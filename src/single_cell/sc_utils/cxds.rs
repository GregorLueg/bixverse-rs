//! Implements the cxds method from Bais & Kostka, Bioinformatics,
//! 2020. The idea: heterotypic doublets co-express marker genes
//! that normally belong to different cell types. Gene pairs whose
//! co-expression is rarer than expected under independence receive
//! high scores. Cells that co-express many such unusual pairs are
//! likely doublets.

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::prelude::*;

////////////
// Consts //
////////////

/// Default number of genes to select for cxds scoring.
pub const CXDS_NTOP: usize = 500;

///////////
// Types //
///////////

/// Per-cell expressed gene set, indexed into the cxds gene set.
///
/// Stored as a sorted `Vec<u16>` since the gene set is at most
/// `CXDS_NTOP` genes wide.
type CellGeneSet = Vec<u16>;

/////////////
// Helpers //
/////////////

/// Approximate the standard normal CDF.
///
/// Uses the Abramowitz & Stegun rational approximation (formula
/// 26.2.17) with accuracy ~1.5e-7.
///
/// ### Params
///
/// * `x` - Quantile.
///
/// ### Returns
///
/// P(Z <= x) for Z ~ N(0, 1).
fn norm_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }
    let t = 1.0 / (1.0 + 0.231_641_9 * x.abs());
    let poly = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let pdf = (-x * x / 2.0).exp() * std::f64::consts::FRAC_2_SQRT_PI * 0.5;
    let tail = pdf * poly;
    if x >= 0.0 { 1.0 - tail } else { tail }
}

/// Compute `-log(P(X >= k))` where `X ~ Binomial(n, p)`.
///
/// Uses the normal approximation with continuity correction, which is accurate
/// for `n * p * (1 - p) >= 5`. For degenerate cases (zero variance), returns
/// 0.0.
///
/// ### Params
///
/// * `k` - Observed count.
/// * `n` - Number of trials.
/// * `p` - Success probability under the null.
///
/// ### Returns
///
/// Negative log upper-tail p-value, clamped to `[0, 700]`.
fn neg_log_binom_upper(k: u32, n: u32, p: f64) -> f32 {
    let np = n as f64 * p;
    let var = np * (1.0 - p);
    if var < 1e-10 {
        return 0.0;
    }
    // Continuity correction: P(X >= k) ~= P(Z >= (k - 0.5 - np) / sd)
    let z = (k as f64 - 0.5 - np) / var.sqrt();
    let tail_p = 1.0 - norm_cdf(z);
    if tail_p < 1e-300 {
        return 700.0; // cap
    }
    (-tail_p.ln()) as f32
}

//////////
// cxds //
//////////

/// Precomputed cxds gene-pair score model.
///
/// Built from observed cells, then applied to both observed and
/// simulated cells to produce per-cell cxds doublet scores.
#[allow(dead_code)]
pub struct CxdsModel {
    /// Gene-pair score matrix, `ntop x ntop`, stored row-major.
    ///
    /// `pair_scores[i * ntop + j]` holds `-log(P)` for the gene
    /// pair `(i, j)`. Symmetric; only `i < j` entries are nonzero.
    pair_scores: Vec<f32>,
    /// Original gene indices of the selected cxds genes (into the
    /// full gene space).
    selected_genes: Vec<usize>,
    /// Number of selected genes.
    ntop: usize,
    /// Reverse lookup: original gene index -> position in
    /// `selected_genes`.
    gene_map: FxHashMap<usize, u16>,
}

impl CxdsModel {
    /// Build the cxds model from observed cells.
    ///
    /// Single-pass approach: reads each cell from disk once, buffering
    /// the expressed HVG indices in memory. After gene selection, the
    /// buffered indices are filtered to the selected set and used for
    /// parallel co-expression counting.
    ///
    /// ### Params
    ///
    /// * `f_path_cell` - Path to the cell-based binary file (CSR).
    /// * `cells_to_keep` - Observed cell indices.
    /// * `hvg_genes` - Highly variable gene indices (the candidate
    ///   pool for cxds gene selection).
    /// * `ntop` - Number of genes to select.
    ///
    /// ### Returns
    ///
    /// `(model, obs_gene_sets)` where `obs_gene_sets[i]` is the
    /// expressed gene set for observed cell `i`.
    pub fn fit(
        f_path_cell: &str,
        cells_to_keep: &[usize],
        hvg_genes: &[usize],
        ntop: usize,
    ) -> (Self, Vec<CellGeneSet>) {
        let n_cells = cells_to_keep.len();
        let hvg_set: FxHashSet<usize> = hvg_genes.iter().copied().collect();
        let reader = ParallelSparseReader::new(f_path_cell).unwrap();

        let mut gene_counts: FxHashMap<usize, u32> = FxHashMap::default();
        let mut cell_hvg_indices: Vec<Vec<usize>> = Vec::with_capacity(n_cells);

        for &cell_idx in cells_to_keep {
            let chunk = reader.read_cell(cell_idx);
            let mut expressed_hvgs = Vec::new();

            for (i, &gene_idx) in chunk.indices.iter().enumerate() {
                let gi = gene_idx as usize;
                if hvg_set.contains(&gi) && chunk.data_raw.get(i) > 0 {
                    *gene_counts.entry(gi).or_insert(0) += 1;
                    expressed_hvgs.push(gi);
                }
            }

            cell_hvg_indices.push(expressed_hvgs);
        }

        // gene selection: top ntop by binomial variance p*(1-p)
        let nf = n_cells as f64;
        let mut gene_vars: Vec<(usize, f64)> = gene_counts
            .iter()
            .map(|(&gene, &count)| {
                let p = count as f64 / nf;
                (gene, p * (1.0 - p))
            })
            .collect();
        gene_vars.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let actual_ntop = ntop.min(gene_vars.len());
        let selected_genes: Vec<usize> = gene_vars[..actual_ntop].iter().map(|&(g, _)| g).collect();

        let gene_map: FxHashMap<usize, u16> = selected_genes
            .iter()
            .enumerate()
            .map(|(pos, &g)| (g, pos as u16))
            .collect();

        let marginals: Vec<f64> = selected_genes
            .iter()
            .map(|g| *gene_counts.get(g).unwrap_or(&0) as f64 / nf)
            .collect();

        // build per-cell gene sets from buffered HVG indices
        let obs_gene_sets: Vec<CellGeneSet> = cell_hvg_indices
            .par_iter()
            .map(|hvgs| {
                let mut expressed: CellGeneSet = hvgs
                    .iter()
                    .filter_map(|&gi| gene_map.get(&gi).copied())
                    .collect();
                expressed.sort_unstable();
                expressed
            })
            .collect();

        // drop the buffered HVG indices; no longer needed -> force compiler to
        // liberate memory
        drop(cell_hvg_indices);

        // parallel co-expression counting
        let n_coexpress = obs_gene_sets
            .par_iter()
            .fold(
                || vec![0u32; actual_ntop * actual_ntop],
                |mut local_coex, expressed| {
                    let ne = expressed.len();
                    for a in 0..ne {
                        let i = expressed[a] as usize;
                        let row = i * actual_ntop;
                        for b in (a + 1)..ne {
                            let j = expressed[b] as usize;
                            local_coex[row + j] += 1;
                        }
                    }
                    local_coex
                },
            )
            .reduce(
                || vec![0u32; actual_ntop * actual_ntop],
                |mut a, b| {
                    for i in 0..a.len() {
                        a[i] += b[i];
                    }
                    a
                },
            );

        // build gene-pair score matrix
        //
        // n_exclusive[i][j] = n_express[i] + n_express[j] - 2 * n_coexpress[i][j]
        let n_express: Vec<u32> = selected_genes
            .iter()
            .map(|g| *gene_counts.get(g).unwrap_or(&0))
            .collect();

        let n = n_cells as u32;
        let mut pair_scores = vec![0.0f32; actual_ntop * actual_ntop];

        for i in 0..actual_ntop {
            for j in (i + 1)..actual_ntop {
                let pi = marginals[i];
                let pj = marginals[j];
                let p_excl = pi * (1.0 - pj) + (1.0 - pi) * pj;
                let coex = n_coexpress[i * actual_ntop + j];
                let k = n_express[i] + n_express[j] - 2 * coex;
                let score = neg_log_binom_upper(k, n, p_excl);
                pair_scores[i * actual_ntop + j] = score;
                pair_scores[j * actual_ntop + i] = score;
            }
        }

        let model = CxdsModel {
            pair_scores,
            selected_genes,
            ntop: actual_ntop,
            gene_map,
        };

        (model, obs_gene_sets)
    }

    /// Compute cxds scores for cells given their expressed gene sets.
    ///
    /// For each cell, sums the pair scores `S[i][j]` over all pairs
    /// `(i, j)` where both genes `i` and `j` are expressed in the
    /// cell. Cells that co-express unusual gene combinations receive
    /// high scores.
    ///
    /// ### Params
    ///
    /// * `gene_sets` - Per-cell expressed gene sets (indices into
    ///   the cxds gene set).
    ///
    /// ### Returns
    ///
    /// Per-cell cxds scores.
    pub fn score(&self, gene_sets: &[CellGeneSet]) -> Vec<f32> {
        gene_sets
            .par_iter()
            .map(|expressed| {
                let mut total = 0.0f32;
                let ne = expressed.len();
                for a in 0..ne {
                    let i = expressed[a] as usize;
                    let row = i * self.ntop;
                    for b in (a + 1)..ne {
                        let j = expressed[b] as usize;
                        total += self.pair_scores[row + j];
                    }
                }
                total
            })
            .collect()
    }

    /// Compute cxds scores for simulated doublets.
    ///
    /// Each simulated doublet is the union of two parent cells'
    /// expressed gene sets (a gene is expressed if either parent
    /// expresses it).
    ///
    /// ### Params
    ///
    /// * `pairs` - Parent cell index pairs `(a, b)` into the
    ///   observed cell set.
    /// * `obs_gene_sets` - Per-observed-cell expressed gene sets.
    ///
    /// ### Returns
    ///
    /// Per-doublet cxds scores.
    pub fn score_simulated(
        &self,
        pairs: &[(usize, usize)],
        obs_gene_sets: &[CellGeneSet],
    ) -> Vec<f32> {
        pairs
            .par_iter()
            .map(|&(a, b)| {
                // Union of both parents' gene sets
                let sa = &obs_gene_sets[a];
                let sb = &obs_gene_sets[b];
                let mut merged = Vec::with_capacity(sa.len() + sb.len());
                let (mut ia, mut ib) = (0, 0);
                while ia < sa.len() && ib < sb.len() {
                    match sa[ia].cmp(&sb[ib]) {
                        std::cmp::Ordering::Less => {
                            merged.push(sa[ia]);
                            ia += 1;
                        }
                        std::cmp::Ordering::Greater => {
                            merged.push(sb[ib]);
                            ib += 1;
                        }
                        std::cmp::Ordering::Equal => {
                            merged.push(sa[ia]);
                            ia += 1;
                            ib += 1;
                        }
                    }
                }
                merged.extend_from_slice(&sa[ia..]);
                merged.extend_from_slice(&sb[ib..]);

                // Score the merged set
                let mut total = 0.0f32;
                let ne = merged.len();
                for a in 0..ne {
                    let i = merged[a] as usize;
                    let row = i * self.ntop;
                    for b in (a + 1)..ne {
                        let j = merged[b] as usize;
                        total += self.pair_scores[row + j];
                    }
                }
                total
            })
            .collect()
    }
}
