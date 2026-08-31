//! DIALOGUE stage one: penalised matrix decomposition.
//!
//! Collapse each cell type's features to one row per sample, drop features that
//! do not vary across samples, keep the samples every cell type shares, and run
//! sparse multi-CCA across the resulting blocks. The weights come back to the
//! cell level as programme scores, and correlating those against expression
//! gives each programme a first, provisional gene signature per cell type.
//!
//! A permutation null over the whole decomposition decides which programmes are
//! real and which cell types each one spans.

use faer::{Mat, MatRef};
use rayon::prelude::*;
use rustc_hash::FxHashSet;

use crate::core::math::linear_algebra::ols_residualise;
use crate::core::math::stats::{
    one_way_anova, p_adjust_fdr, partial_correlation, wilcox_rank_sum_greater_approx,
};
use crate::core::math::vector_helpers::{
    median, pearson_correlation, quantile_sorted, rank_vector,
};
use crate::methods::multi_cca::{
    MultiCcaParams, MultiCcaPermuteParams, multi_cca, multi_cca_permute,
};
use crate::prelude::*;
use crate::single_cell::sc_analysis::dialogue::params::{
    Averaging, DialogueParams, MIN_ANOVA_FEATURES, MIN_SHARED_SAMPLES,
};

////////////
// Consts //
////////////

/// Genes read per batch during the correlation sweeps.
const GENE_BATCH: usize = 2000;

//////////////////
// Result types //
//////////////////

/// Up and down gene lists for one programme in one cell type.
#[derive(Clone, Debug, Default)]
pub struct ProgrammeSignature {
    /// Genes positively correlated with the programme score.
    pub up: Vec<usize>,
    /// Genes negatively correlated with it.
    pub down: Vec<usize>,
}

impl ProgrammeSignature {
    /// Every gene in the signature, both directions, sorted and deduplicated.
    ///
    /// ### Returns
    ///
    /// The union of `up` and `down`.
    pub fn all(&self) -> Vec<usize> {
        let mut out: Vec<usize> = self.up.iter().chain(self.down.iter()).copied().collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// What stage one produced.
#[derive(Clone, Debug)]
pub struct DialogueStep1Result {
    /// Feature columns surviving the ANOVA filter, per cell type.
    pub kept_features: Vec<Vec<usize>>,
    /// Sample codes present in every cell type, ascending.
    pub shared_samples: Vec<usize>,
    /// Sparse canonical weights per cell type, `kept_features x k`.
    pub ws: Vec<Mat<f64>>,
    /// Programme scores per cell type, `n_cells_in_type x k`, already
    /// residualised on the cell quality covariate.
    pub scores: Vec<Mat<f64>>,
    /// The same projections *before* residualisation. Stage three refits
    /// against these rather than `scores`; see `run_step3` for why.
    pub raw_scores: Vec<Mat<f64>>,
    /// Empirical p-value per programme and cell-type pair, `k x n_pairs`.
    pub emp_p: Mat<f64>,
    /// The real fit's canonical correlation per programme and pair.
    pub pair_cor: Mat<f64>,
    /// Which cell types each programme spans.
    pub mcp_cell_types: Vec<Vec<usize>>,
    /// Provisional signatures, indexed `[cell_type][programme]`.
    pub signatures: Vec<Vec<ProgrammeSignature>>,
}

//////////////////
// CellTypeView //
//////////////////

/// One cell type's inputs, resolved against the global index space.
#[derive(Clone, Debug)]
pub(crate) struct CellTypeView {
    /// Global cell indices belonging to this cell type.
    pub cells: Vec<usize>,
    /// Cell quality per cell, parallel to `cells`.
    pub quality: Vec<f64>,
    /// Distinct sample codes present, ascending.
    pub samples: Vec<usize>,
    /// Row positions into `cells` grouped by sample, parallel to `samples`.
    pub rows_by_sample: Vec<Vec<usize>>,
}

impl CellTypeView {
    /// Resolves one cell type against the global per-cell metadata.
    ///
    /// ### Params
    ///
    /// * `cells` - Global cell indices for this cell type
    /// * `sample_ids` - Sample code per global cell
    /// * `quality` - Quality covariate per global cell
    ///
    /// ### Returns
    ///
    /// The view, or [BixverseErrors::InvalidArgument] for an out-of-range cell.
    pub(crate) fn new(
        cells: &[usize],
        sample_ids: &[usize],
        quality: &[f64],
    ) -> Result<Self, BixverseErrors> {
        let mut sample_of_cell = Vec::with_capacity(cells.len());
        let mut q = Vec::with_capacity(cells.len());
        for &c in cells.iter() {
            if c >= sample_ids.len() || c >= quality.len() {
                return Err(BixverseErrors::InvalidArgument(format!(
                    "cell index {c} is outside the per-cell metadata."
                )));
            }
            sample_of_cell.push(sample_ids[c]);
            q.push(quality[c]);
        }

        let mut samples: Vec<usize> = sample_of_cell.clone();
        samples.sort_unstable();
        samples.dedup();
        let mut rows_by_sample: Vec<Vec<usize>> = vec![Vec::new(); samples.len()];
        for (row, s) in sample_of_cell.iter().enumerate() {
            let slot = samples.binary_search(s).expect("sample was just collected");
            rows_by_sample[slot].push(row);
        }

        Ok(Self {
            cells: cells.to_vec(),
            quality: q,
            samples,
            rows_by_sample,
        })
    }

    /// Dense map from global cell index to row position within this cell type.
    ///
    /// `u32::MAX` marks a cell belonging to some other cell type. Dense rather
    /// than a hash map because this is looked up once per non-zero entry in
    /// every gene-major sweep, where a hash would dominate.
    ///
    /// ### Params
    ///
    /// * `total_cells` - Cells in the store, from the reader header
    ///
    /// ### Returns
    ///
    /// A table of length `total_cells`.
    pub(crate) fn position_table(&self, total_cells: usize) -> Vec<u32> {
        let mut position = vec![u32::MAX; total_cells];
        for (row, &c) in self.cells.iter().enumerate() {
            position[c] = row as u32;
        }
        position
    }
}

////////////////////////
// Sample aggregation //
////////////////////////

/// Collapses cell-level features to one row per sample.
///
/// Row order is ascending sample code. Upstream orders rows by descending
/// sample frequency instead, which reads as arbitrary; it is, and it does not
/// matter. The cross products the decomposition is built from are sums over
/// rows, so any permutation applied to every block alike leaves them unchanged,
/// and the column standardisation and winsorising that follow are per column.
///
/// ### Params
///
/// * `features` - Cell-level features, `n_cells_in_type x p`
/// * `view` - The cell type's sample grouping
/// * `averaging` - Median or mean
///
/// ### Returns
///
/// An `n_samples x p` matrix, rows in `view.samples` order.
fn sample_average(features: MatRef<f64>, view: &CellTypeView, averaging: Averaging) -> Mat<f64> {
    let p = features.ncols();
    let n_samples = view.samples.len();
    let mut out = Mat::<f64>::zeros(n_samples, p);
    let mut buffer: Vec<f64> = Vec::new();

    for (slot, rows) in view.rows_by_sample.iter().enumerate() {
        for j in 0..p {
            buffer.clear();
            buffer.extend(rows.iter().map(|&r| features[(r, j)]));
            out[(slot, j)] = match averaging {
                Averaging::Median => median(&buffer).unwrap_or(f64::NAN),
                Averaging::Mean => buffer.iter().sum::<f64>() / buffer.len() as f64,
            };
        }
    }
    out
}

/// Keeps only features that vary across samples.
///
/// One-way ANOVA of each cell-level feature against the sample it came from,
/// restricted to samples contributing at least `abn_c` cells, then Benjamini-
/// Hochberg across features.
///
/// ### Params
///
/// * `features` - Cell-level features, `n_cells_in_type x p`
/// * `view` - The cell type's sample grouping
/// * `cell_type` - Index of the cell type, for the error message
/// * `abn_c` - Minimum cells a sample must contribute to be tested
/// * `p_anova` - Adjusted p below which a feature is kept
///
/// ### Returns
///
/// The surviving column indices, [BixverseErrors::DialogueTooFewFeatures] when
/// too few survive, or [BixverseErrors::DialogueTooFewAbundantSamples] when
/// too few samples carry enough cells to test against.
fn anova_filter(
    features: MatRef<f64>,
    view: &CellTypeView,
    cell_type: usize,
    abn_c: usize,
    p_anova: f64,
) -> Result<Vec<usize>, BixverseErrors> {
    let p = features.ncols();
    // Only the abundant samples take part, and they are recoded densely so the
    // ANOVA does not count the dropped ones as empty groups.
    let mut recode = vec![usize::MAX; view.samples.len()];
    let mut n_kept_samples = 0;
    for (slot, rows) in view.rows_by_sample.iter().enumerate() {
        if rows.len() >= abn_c {
            recode[slot] = n_kept_samples;
            n_kept_samples += 1;
        }
    }
    if n_kept_samples < 2 {
        return Err(BixverseErrors::DialogueTooFewAbundantSamples {
            cell_type,
            found: n_kept_samples,
            min_cells: abn_c,
        });
    }

    let mut rows: Vec<usize> = Vec::new();
    let mut groups: Vec<usize> = Vec::new();
    for (slot, members) in view.rows_by_sample.iter().enumerate() {
        if recode[slot] == usize::MAX {
            continue;
        }
        for &r in members.iter() {
            rows.push(r);
            groups.push(recode[slot]);
        }
    }

    let pvals: Vec<f64> = (0..p)
        .into_par_iter()
        .map(|j| {
            let values: Vec<f64> = rows.iter().map(|&r| features[(r, j)]).collect();
            one_way_anova(&values, &groups, n_kept_samples)
                .map(|(_, p)| p)
                .unwrap_or(1.0)
        })
        .collect();

    let adjusted = p_adjust_fdr(&pvals);
    let kept: Vec<usize> = (0..p).filter(|&j| adjusted[j] < p_anova).collect();
    if kept.len() < MIN_ANOVA_FEATURES {
        return Err(BixverseErrors::DialogueTooFewFeatures {
            cell_type,
            found: kept.len(),
            total: p,
            minimum: MIN_ANOVA_FEATURES,
        });
    }
    Ok(kept)
}

/// Column-standardises and winsorises a sample-level matrix.
///
/// Centres each column, divides by its standard deviation with an `n - 1`
/// denominator, then clips to the `cap` and `1 - cap` quantiles under R's type
/// 7 interpolation. Both steps see every row the caller passes; see
/// [run_step1] for why that matters.
///
/// ### Params
///
/// * `mat` - Sample-level features
/// * `cap` - Winsorising tail fraction. Zero skips the clipping.
///
/// ### Returns
///
/// The transformed matrix.
fn standardise_and_cap(mat: MatRef<f64>, cap: f64) -> Mat<f64> {
    let n = mat.nrows();
    let p = mat.ncols();
    let mut out = Mat::<f64>::zeros(n, p);

    for j in 0..p {
        let col: Vec<f64> = (0..n).map(|i| mat[(i, j)]).collect();
        let mean = col.iter().sum::<f64>() / n as f64;
        let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
        let sd = var.sqrt();
        let sd = if sd > 0.0 && sd.is_finite() { sd } else { 1.0 };
        for i in 0..n {
            out[(i, j)] = (col[i] - mean) / sd;
        }
        if cap <= 0.0 {
            continue;
        }
        let mut scaled: Vec<f64> = (0..n).map(|i| out[(i, j)]).collect();
        scaled.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo = quantile_sorted(&scaled, cap);
        let hi = quantile_sorted(&scaled, 1.0 - cap);
        for i in 0..n {
            out[(i, j)] = out[(i, j)].clamp(lo, hi);
        }
    }
    out
}

//////////////////////////
// The permutation null //
//////////////////////////

/// Every unordered cell-type pair, in `(0,1), (0,2), ..., (1,2), ...` order.
///
/// ### Params
///
/// * `k` - Number of cell types
///
/// ### Returns
///
/// The pairs, matching R's `combn(.., 2)` column order.
fn cell_type_pairs(k: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(k * (k - 1) / 2);
    for i in 0..k {
        for j in (i + 1)..k {
            out.push((i, j));
        }
    }
    out
}

/// Canonical correlation per programme and pair for one set of weights.
///
/// The variates are formed from the blocks as the caller supplies them, not
/// from the internally re-standardised copies the decomposition works on. That
/// is what upstream does, and the two differ because winsorising leaves the
/// column standard deviations slightly below one.
///
/// ### Params
///
/// * `blocks` - Sample-level matrices per cell type
/// * `ws` - Weights per cell type, `p_k x n_components`
/// * `pairs` - Cell-type pairs
///
/// ### Returns
///
/// An `n_components x n_pairs` matrix of correlations.
fn pairwise_canonical_cor(
    blocks: &[Mat<f64>],
    ws: &[Mat<f64>],
    pairs: &[(usize, usize)],
) -> Mat<f64> {
    let k = ws[0].ncols();
    let variates: Vec<Mat<f64>> = blocks.iter().zip(ws.iter()).map(|(b, w)| b * w).collect();
    let mut out = Mat::<f64>::zeros(k, pairs.len());
    for (col, &(a, b)) in pairs.iter().enumerate() {
        for comp in 0..k {
            let va: Vec<f64> = (0..variates[a].nrows())
                .map(|i| variates[a][(i, comp)])
                .collect();
            let vb: Vec<f64> = (0..variates[b].nrows())
                .map(|i| variates[b][(i, comp)])
                .collect();
            out[(comp, col)] = pearson_correlation(&va, &vb).unwrap_or(0.0);
        }
    }
    out
}

/// Empirical p-value per programme and cell-type pair.
///
/// The real decomposition is scored against `n_permutations - 1` refits on data
/// whose rows have been shuffled independently within every column of every
/// block, which destroys the cross-block association while leaving each block's
/// own column distributions alone.
///
/// The test is a one-sided rank sum of the single real value against the null,
/// as upstream. Note what that implies for the floor: with 99 nulls R takes the
/// normal approximation rather than the exact distribution, so the smallest
/// attainable p is about 0.045, not 0.01. The programme-to-cell-type threshold
/// sits at 0.1, inside that gap.
///
/// ### Params
///
/// * `blocks` - Sample-level matrices per cell type
/// * `cca_params` - Parameters for each refit
/// * `n_permutations` - Total runs, one real and the rest permuted
/// * `pairs` - Cell-type pairs
/// * `seed` - Seed for the shuffles
///
/// ### Returns
///
/// `(empirical_p, real_correlations)`, both `k x n_pairs`.
fn empirical_pvalues(
    blocks: &[Mat<f64>],
    cca_params: &MultiCcaParams<f64>,
    n_permutations: usize,
    pairs: &[(usize, usize)],
    seed: u64,
) -> Result<(Mat<f64>, Mat<f64>), BixverseErrors> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;

    let refs: Vec<MatRef<f64>> = blocks.iter().map(|m| m.as_ref()).collect();
    let real_fit = multi_cca(&refs, Some(cca_params.clone()))?;
    let real = pairwise_canonical_cor(blocks, &real_fit.ws, pairs);

    let k = cca_params.n_components;
    let n_null = n_permutations.saturating_sub(1);
    let n = blocks[0].nrows();

    // Draw every permutation up front so the parallel refits stay reproducible
    // regardless of how rayon schedules them.
    let mut rng = StdRng::seed_from_u64(seed);
    let orders: Vec<Vec<Vec<usize>>> = (0..n_null)
        .map(|_| {
            blocks
                .iter()
                .map(|b| {
                    (0..b.ncols())
                        .flat_map(|_| {
                            let mut o: Vec<usize> = (0..n).collect();
                            o.shuffle(&mut rng);
                            o
                        })
                        .collect::<Vec<usize>>()
                })
                .collect()
        })
        .collect();

    let null: Vec<Mat<f64>> = orders
        .par_iter()
        .map(|order| {
            let shuffled: Vec<Mat<f64>> = blocks
                .iter()
                .zip(order.iter())
                .map(|(b, o)| Mat::<f64>::from_fn(n, b.ncols(), |i, j| b[(o[j * n + i], j)]))
                .collect();
            let refs: Vec<MatRef<f64>> = shuffled.iter().map(|m| m.as_ref()).collect();
            match multi_cca(&refs, Some(cca_params.clone())) {
                Ok(fit) => pairwise_canonical_cor(&shuffled, &fit.ws, pairs),
                Err(_) => Mat::<f64>::zeros(k, pairs.len()),
            }
        })
        .collect();

    let mut emp_p = Mat::<f64>::zeros(k, pairs.len());
    for comp in 0..k {
        for col in 0..pairs.len() {
            let observed = [real[(comp, col)]];
            let nulls: Vec<f64> = null.iter().map(|m| m[(comp, col)]).collect();
            emp_p[(comp, col)] =
                wilcox_rank_sum_greater_approx(&observed, &nulls).unwrap_or(f64::NAN);
        }
    }

    Ok((emp_p, real))
}

/// Decides which cell types each programme spans.
///
/// Greedy peeling on the symmetrised empirical p-values: drop the least
/// connected cell type until every survivor is connected to at least half of
/// them. A cell type counts as connected to itself, which is what putting 0.05
/// on the diagonal achieves upstream.
///
/// With exactly two cell types there is nothing to peel and the programme
/// either spans both or neither.
///
/// ### Params
///
/// * `emp_p` - Empirical p-values, `k x n_pairs`
/// * `pairs` - Cell-type pairs matching the columns
/// * `n_cell_types` - Number of cell types
/// * `threshold` - p below which a pair counts as connected
///
/// ### Returns
///
/// The cell types spanned, per programme.
fn assign_cell_types(
    emp_p: MatRef<f64>,
    pairs: &[(usize, usize)],
    n_cell_types: usize,
    threshold: f64,
) -> Vec<Vec<usize>> {
    let k = emp_p.nrows();
    let mut out = Vec::with_capacity(k);

    for comp in 0..k {
        if n_cell_types == 2 {
            out.push(if emp_p[(comp, 0)] < threshold {
                vec![0, 1]
            } else {
                Vec::new()
            });
            continue;
        }

        // Symmetric connectivity, self included.
        let mut connected = vec![false; n_cell_types * n_cell_types];
        for i in 0..n_cell_types {
            connected[i * n_cell_types + i] = true;
        }
        for (col, &(a, b)) in pairs.iter().enumerate() {
            let hit = emp_p[(comp, col)] < threshold;
            connected[a * n_cell_types + b] = hit;
            connected[b * n_cell_types + a] = hit;
        }

        let mut survivors: Vec<usize> = (0..n_cell_types).collect();
        loop {
            let degrees: Vec<usize> = survivors
                .iter()
                .map(|&i| {
                    survivors
                        .iter()
                        .filter(|&&j| connected[i * n_cell_types + j])
                        .count()
                })
                .collect();
            let lowest = degrees.iter().copied().min().unwrap_or(0);
            let needed = survivors.len().div_ceil(2);
            if lowest >= needed || survivors.len() <= 1 {
                break;
            }
            // Drop the first survivor sitting at the minimum, matching
            // upstream's `[1]` on the candidate set.
            let victim = degrees
                .iter()
                .position(|&d| d <= lowest)
                .expect("the minimum is attained");
            survivors.remove(victim);
        }
        out.push(survivors);
    }
    out
}

/////////////////////////////
// Gene-level correlations //
/////////////////////////////

/// Per-column sums needed to correlate a sparse gene against dense scores.
struct ScoreMoments {
    /// Number of cells.
    n: f64,
    /// Column sums.
    sum: Vec<f64>,
    /// Column sums of squares.
    sum_sq: Vec<f64>,
    /// The scores themselves, column-major for the sparse gather.
    columns: Vec<Vec<f64>>,
}

impl ScoreMoments {
    /// Precomputes what every gene's correlation reuses.
    ///
    /// ### Params
    ///
    /// * `scores` - Programme scores, `n_cells x k`
    ///
    /// ### Returns
    ///
    /// The moments.
    fn new(scores: MatRef<f64>) -> Self {
        let n = scores.nrows();
        let k = scores.ncols();
        let mut columns = Vec::with_capacity(k);
        let mut sum = Vec::with_capacity(k);
        let mut sum_sq = Vec::with_capacity(k);
        for j in 0..k {
            let col: Vec<f64> = (0..n).map(|i| scores[(i, j)]).collect();
            sum.push(col.iter().sum());
            sum_sq.push(col.iter().map(|v| v * v).sum());
            columns.push(col);
        }
        Self {
            n: n as f64,
            sum,
            sum_sq,
            columns,
        }
    }
}

/// Pearson correlation of one sparse gene against every programme score.
///
/// The gene is given as its non-zero entries only. The raw-moment form is used
/// deliberately here: it is the only way to avoid densifying, the scores are
/// residualised so they sit near zero, and expression is non-negative and
/// small, so the cancellation the centred form guards against does not arise.
///
/// ### Params
///
/// * `positions` - Row positions of the non-zero entries
/// * `values` - The non-zero values
/// * `moments` - Precomputed score moments
///
/// ### Returns
///
/// One correlation per programme.
fn sparse_gene_correlations(
    positions: &[usize],
    values: &[f64],
    moments: &ScoreMoments,
) -> Vec<f64> {
    let n = moments.n;
    let sum_x: f64 = values.iter().sum();
    let sum_xx: f64 = values.iter().map(|v| v * v).sum();
    let var_x = sum_xx - sum_x * sum_x / n;
    // A gene with no variation has no correlation; R's `cor` gives NA and
    // `get.top.elements` drops it. Returning 0.0 would leave it in the vector
    // the cutoffs are computed from.
    if var_x <= 0.0 {
        return vec![f64::NAN; moments.columns.len()];
    }

    moments
        .columns
        .iter()
        .enumerate()
        .map(|(j, col)| {
            let sum_xy: f64 = positions
                .iter()
                .zip(values.iter())
                .map(|(&p, v)| col[p] * v)
                .sum();
            let cov = sum_xy - sum_x * moments.sum[j] / n;
            let var_y = moments.sum_sq[j] - moments.sum[j] * moments.sum[j] / n;
            if var_y <= 0.0 {
                return f64::NAN;
            }
            let r = cov / (var_x * var_y).sqrt();
            if r.is_finite() {
                r.clamp(-1.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

/// Top correlated genes in each direction.
///
/// Reproduces `get.top.cor`: up to `q` genes on each side, but only those past
/// `min_ci`, and ties at the cutoff are all kept, so a list can come back
/// longer than `q`.
///
/// ### Params
///
/// * `values` - Correlation per gene
/// * `gene_ids` - Gene identifier per entry
/// * `q` - Target list length
/// * `min_ci` - Minimum absolute correlation
///
/// ### Returns
///
/// The up and down lists, each sorted by gene identifier.
fn top_correlated(values: &[f64], gene_ids: &[usize], q: usize, min_ci: f64) -> ProgrammeSignature {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() || q == 0 {
        return ProgrammeSignature::default();
    }
    let mut sorted = finite;
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = q.min(sorted.len());

    // Descending for the up side: the m-th largest, floored at min_ci.
    let up_cut = sorted[sorted.len() - m].max(min_ci);
    // Ascending for the down side: the m-th smallest, capped at -min_ci.
    let down_cut = sorted[m - 1].min(-min_ci);

    let mut up: Vec<usize> = Vec::new();
    let mut down: Vec<usize> = Vec::new();
    for (v, &g) in values.iter().zip(gene_ids.iter()) {
        if !v.is_finite() {
            continue;
        }
        if *v >= up_cut {
            up.push(g);
        }
        if *v <= down_cut {
            down.push(g);
        }
    }
    up.sort_unstable();
    down.sort_unstable();
    ProgrammeSignature { up, down }
}

/// Builds each programme's provisional signature for one cell type.
///
/// Two passes over the genes. The first correlates every gene against the
/// scores and shortlists the strongest in each direction; the second re-ranks
/// only that shortlist by Spearman partial correlation, holding the cell
/// quality covariate fixed, and drops anything that fails a Bonferroni cut
/// across the full gene list.
///
/// ### Params
///
/// * `reader` - Gene-major reader
/// * `genes` - Genes to consider
/// * `view` - The cell type
/// * `scores` - Programme scores, `n_cells_in_type x k`
/// * `n_genes` - Target signature length per direction
/// * `min_ci` - Minimum absolute correlation
///
/// ### Returns
///
/// One [ProgrammeSignature] per programme.
fn build_signatures<S: SingleCellReading>(
    reader: &S,
    genes: &[usize],
    view: &CellTypeView,
    scores: MatRef<f64>,
    n_genes: usize,
    min_ci: f64,
) -> Result<Vec<ProgrammeSignature>, BixverseErrors> {
    let k = scores.ncols();
    let n_in_type = view.cells.len();
    let total_cells = reader.get_header().total_cells;

    let position = view.position_table(total_cells);

    let moments = ScoreMoments::new(scores);

    // Pass one: Pearson correlation of every gene against every programme.
    let mut all_cor: Vec<Vec<f64>> = Vec::with_capacity(genes.len());
    for batch in genes.chunks(GENE_BATCH) {
        let chunks = reader.read_gene_parallel(batch)?;
        let batch_cor: Vec<Vec<f64>> = chunks
            .par_iter()
            .map(|chunk| {
                let mut positions = Vec::with_capacity(chunk.indices.len());
                let mut values = Vec::with_capacity(chunk.indices.len());
                for (&cell, value) in chunk.indices.iter().zip(chunk.data_norm.iter()) {
                    let slot = position[cell as usize];
                    if slot != u32::MAX {
                        positions.push(slot as usize);
                        values.push(value.to_f32() as f64);
                    }
                }
                sparse_gene_correlations(&positions, &values, &moments)
            })
            .collect();
        all_cor.extend(batch_cor);
    }

    // Shortlist: the union over programmes of the top genes in each direction.
    let mut shortlist: FxHashSet<usize> = FxHashSet::default();
    for comp in 0..k {
        let column: Vec<f64> = all_cor.iter().map(|c| c[comp]).collect();
        let sig = top_correlated(&column, genes, n_genes, min_ci);
        shortlist.extend(sig.up.iter().copied());
        shortlist.extend(sig.down.iter().copied());
    }
    let mut shortlist: Vec<usize> = shortlist.into_iter().collect();
    shortlist.sort_unstable();

    if shortlist.is_empty() {
        return Ok(vec![ProgrammeSignature::default(); k]);
    }

    // Pass two: Spearman partial correlation on the shortlist, controlling for
    // cell quality. Ranking the scores and the covariate once here is what
    // keeps this affordable; only the gene has to be ranked per iteration.
    let ranked_quality = rank_vector(&view.quality);
    let ranked_scores: Vec<Vec<f64>> = (0..k)
        .map(|j| {
            let col: Vec<f64> = (0..n_in_type).map(|i| scores[(i, j)]).collect();
            rank_vector(&col)
        })
        .collect();
    // Upstream's cut is Bonferroni over every gene it was given, not over the
    // shortlist.
    let bonferroni = 0.05 / genes.len() as f64;

    let mut partial: Vec<Vec<f64>> = Vec::with_capacity(shortlist.len());
    for batch in shortlist.chunks(GENE_BATCH) {
        let chunks = reader.read_gene_parallel(batch)?;
        let batch_partial: Vec<Vec<f64>> = chunks
            .par_iter()
            .map(|chunk| {
                let mut dense = vec![0.0_f64; n_in_type];
                for (&cell, value) in chunk.indices.iter().zip(chunk.data_norm.iter()) {
                    let slot = position[cell as usize];
                    if slot != u32::MAX {
                        dense[slot as usize] = value.to_f32() as f64;
                    }
                }
                let ranked_gene = rank_vector(&dense);
                (0..k)
                    .map(|j| {
                        // Already ranked, so ask for the Pearson form: the
                        // Spearman path would rank the ranks.
                        match partial_correlation(
                            &ranked_gene,
                            &ranked_scores[j],
                            &ranked_quality,
                            false,
                        ) {
                            Ok((r, p)) if p <= bonferroni => r,
                            _ => 0.0,
                        }
                    })
                    .collect()
            })
            .collect();
        partial.extend(batch_partial);
    }

    Ok((0..k)
        .map(|comp| {
            let column: Vec<f64> = partial.iter().map(|c| c[comp]).collect();
            top_correlated(&column, &shortlist, n_genes, min_ci)
        })
        .collect())
}

/////////////////////////
// DIALOGUE: Stage one //
/////////////////////////

/// Runs DIALOGUE stage one.
///
/// ### Params
///
/// * `reader` - Gene-major reader
/// * `views` - One resolved view per cell type
/// * `features` - Cell-level features per cell type, rows aligned to the views
/// * `genes` - Genes to consider for the signatures
/// * `params` - The full parameter set
/// * `verbose` - `0` silent, non-zero prints progress
///
/// ### Returns
///
/// The [DialogueStep1Result], or the first error encountered.
pub(crate) fn dialogue_step1_run<S: SingleCellReading>(
    reader: &S,
    views: &[CellTypeView],
    features: &[MatRef<f64>],
    genes: &[usize],
    params: &DialogueParams,
    verbose: usize,
) -> Result<DialogueStep1Result, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let n_types = views.len();
    let pmd = &params.pmd;

    // Sample-level matrices, one per cell type, plus the feature filter.
    let mut kept_features = Vec::with_capacity(n_types);
    let mut sample_matrices = Vec::with_capacity(n_types);
    for (i, view) in views.iter().enumerate() {
        let kept = if pmd.spatial {
            (0..features[i].ncols()).collect()
        } else {
            anova_filter(features[i], view, i, pmd.abn_c, pmd.p_anova)?
        };
        if verbosity.normal_verbosity() {
            println!(
                "Cell type {i}: kept {} of {} features.",
                kept.len(),
                features[i].ncols()
            );
        }
        let averaged = sample_average(features[i], view, pmd.averaging);
        let subset =
            Mat::<f64>::from_fn(averaged.nrows(), kept.len(), |r, c| averaged[(r, kept[c])]);
        kept_features.push(kept);
        sample_matrices.push(subset);
    }

    // Samples every cell type has.
    let mut shared: Vec<usize> = views[0].samples.clone();
    for view in views.iter().skip(1) {
        let present: FxHashSet<usize> = view.samples.iter().copied().collect();
        shared.retain(|s| present.contains(s));
    }
    if shared.len() < MIN_SHARED_SAMPLES {
        return Err(BixverseErrors::DialogueTooFewSharedSamples {
            found: shared.len(),
            minimum: MIN_SHARED_SAMPLES,
        });
    }
    if verbosity.normal_verbosity() {
        println!("{} samples shared across all cell types.", shared.len());
    }

    let blocks: Vec<Mat<f64>> = views
        .iter()
        .zip(sample_matrices.iter())
        .map(|(view, mat)| {
            let transformed = if pmd.centre {
                standardise_and_cap(mat.as_ref(), pmd.cap)
            } else {
                mat.clone()
            };
            let rows: Vec<usize> = shared
                .iter()
                .map(|s| view.samples.binary_search(s).expect("shared sample"))
                .collect();
            Mat::<f64>::from_fn(rows.len(), transformed.ncols(), |r, c| {
                transformed[(rows[r], c)]
            })
        })
        .collect();

    let refs: Vec<MatRef<f64>> = blocks.iter().map(|m| m.as_ref()).collect();
    let penalties = if pmd.extra_sparse {
        let tuned = multi_cca_permute(
            &refs,
            Some(MultiCcaPermuteParams {
                seed: pmd.seed,
                ..Default::default()
            }),
        )?;
        tuned.best_penalties
    } else {
        let requested = (blocks[0].ncols() as f64).sqrt() / 2.0;
        let feasible = blocks
            .iter()
            .map(|b| (b.ncols() as f64).sqrt())
            .fold(f64::INFINITY, f64::min);
        let bound = requested.min(feasible).max(1.0);
        if bound < requested && verbosity.normal_verbosity() {
            println!(
                "L1 bound clamped from {requested:.3} to {bound:.3}; the narrowest cell type \
                 cannot support the wider bound."
            );
        }
        vec![bound; n_types]
    };

    let cca_params = MultiCcaParams {
        n_components: pmd.k,
        penalties: Some(penalties.clone()),
        ..Default::default()
    };
    let fit = multi_cca(&refs, Some(cca_params.clone()))?;

    let pairs = cell_type_pairs(n_types);
    let (emp_p, pair_cor) =
        empirical_pvalues(&blocks, &cca_params, pmd.n_permutations, &pairs, pmd.seed)?;
    let mcp_cell_types = assign_cell_types(emp_p.as_ref(), &pairs, n_types, pmd.mcp_assignment_p);

    let mut scores = Vec::with_capacity(n_types);
    let mut raw_scores = Vec::with_capacity(n_types);
    let mut signatures = Vec::with_capacity(n_types);
    for (i, view) in views.iter().enumerate() {
        let kept = &kept_features[i];
        let sub = Mat::<f64>::from_fn(features[i].nrows(), kept.len(), |r, c| {
            features[i][(r, kept[c])]
        });
        let raw_scores_i = &sub * &fit.ws[i];
        let quality = Mat::<f64>::from_fn(view.cells.len(), 1, |r, _| view.quality[r]);
        let residualised = ols_residualise(raw_scores_i.as_ref(), quality.as_ref())?;

        let sig = build_signatures(
            reader,
            genes,
            view,
            residualised.as_ref(),
            pmd.n_genes,
            pmd.min_ci,
        )?;
        if verbosity.normal_verbosity() {
            let sizes: Vec<String> = sig
                .iter()
                .map(|s| format!("{}/{}", s.up.len(), s.down.len()))
                .collect();
            println!(
                "Cell type {i}: signature sizes up/down {}",
                sizes.join(", ")
            );
        }
        raw_scores.push(raw_scores_i);
        scores.push(residualised);
        signatures.push(sig);
    }

    Ok(DialogueStep1Result {
        kept_features,
        shared_samples: shared,
        ws: fit.ws,
        scores,
        raw_scores,
        emp_p,
        pair_cor,
        mcp_cell_types,
        signatures,
    })
}
