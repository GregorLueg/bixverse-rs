//! Dissolve too-small candidate metacells.
//!
//! ### Decision rules per candidate
//!
//! 1. If size `< min_metacell_size`: dissolve. Hard floor.
//! 2. If size `>= min_robust_size` OR umis `>= min_robust_umis`: keep.
//! 3. If `min_convincing_gene_fold_factor` is `None`: keep (no further
//!    test).
//! 4. Otherwise: keep iff at least one gene has log-fold-factor
//!    `>= min_convincing_gene_fold_factor` over the population mean.

use crate::core::math::sparse::CompressedSparseData2;

use super::params::DissolveParams;

/////////////
// Helpers //
/////////////

/// Sum UMI counts per gene across cells matching a predicate.
///
/// ### Params
///
/// * `raw_umis` - CSR matrix of raw UMI counts.
/// * `pred` - Predicate on the candidate assignment; only cells where
///   `pred(candidate_of_cell[i])` is `true` contribute.
/// * `candidate_of_cell` - Candidate assignment per cell.
///
/// ### Returns
///
/// A `Vec<f64>` of length `n_genes` with the summed UMI counts.
fn sum_per_gene(
    raw_umis: &CompressedSparseData2<u32, f32>,
    pred: impl Fn(i32) -> bool,
    candidate_of_cell: &[i32],
) -> Vec<f64> {
    let n_genes = raw_umis.shape.1;
    let mut sums = vec![0.0_f64; n_genes];
    for cell in 0..raw_umis.shape.0 {
        if !pred(candidate_of_cell[cell]) {
            continue;
        }
        let start = raw_umis.indptr[cell];
        let end = raw_umis.indptr[cell + 1];
        for idx in start..end {
            let g = raw_umis.indices[idx];
            sums[g] += raw_umis.data[idx] as f64;
        }
    }
    sums
}

//////////
// Main //
//////////

/// Dissolve candidate metacells. See module-level docs.
///
/// ### Params
///
/// * `raw_umis` - `(n_cells, n_genes)` raw UMI matrix in CSR. Used for
///   the convincing-gene test.
/// * `umis_per_cell` - Pre-summed library size per cell.
/// * `candidate_of_cell` - Per-cell candidate metacell assignment from
///   `compute_candidate_metacells`.
/// * `deviant_of_cell` - Per-cell deviant flag from `find_deviant_cells`.
///   Deviant cells are excluded from their candidate's size and UMI sums
///   AND become outliers in the output.
/// * `target_metacell_size` - Used to compute `min_robust_size`.
/// * `target_metacell_umis` - Used to compute `min_robust_umis`.
/// * `min_metacell_size` - Hard floor on cell count.
/// * `params` - Dissolve parameters.
///
/// ### Returns
///
/// `(metacell_of_cell, dissolved_of_cell)` where `metacell_of_cell` has
/// dense IDs in `[0, k)` for surviving metacells and `-1` elsewhere.
#[allow(clippy::too_many_arguments)]
pub fn dissolve_metacells(
    raw_umis: &CompressedSparseData2<u32, f32>,
    umis_per_cell: &[f32],
    candidate_of_cell: &[i32],
    deviant_of_cell: &[bool],
    target_metacell_size: usize,
    target_metacell_umis: f64,
    min_metacell_size: usize,
    params: &DissolveParams,
) -> (Vec<i32>, Vec<bool>) {
    let n_cells = raw_umis.shape.0;
    let n_genes = raw_umis.shape.1;
    assert_eq!(umis_per_cell.len(), n_cells);
    assert_eq!(candidate_of_cell.len(), n_cells);
    assert_eq!(deviant_of_cell.len(), n_cells);

    let min_robust_size =
        ((target_metacell_size as f32) * params.min_robust_size_factor).floor() as usize;
    let min_robust_umis = (target_metacell_umis * params.min_robust_size_factor as f64).floor();

    let total_per_gene = sum_per_gene(raw_umis, |c| c >= 0, candidate_of_cell);
    let total_population_umis: f64 = total_per_gene.iter().sum();
    let fraction_per_gene: Vec<f64> = if total_population_umis > 0.0 {
        total_per_gene
            .iter()
            .map(|&v| v / total_population_umis)
            .collect()
    } else {
        vec![0.0; n_genes]
    };

    let n_candidates = (candidate_of_cell.iter().copied().max().unwrap_or(-1) + 1) as usize;
    let mut keep = vec![false; n_candidates];

    for c in 0..n_candidates {
        // Cells in this candidate that aren't deviants.
        let cell_indices: Vec<usize> = (0..n_cells)
            .filter(|&i| candidate_of_cell[i] == c as i32 && !deviant_of_cell[i])
            .collect();
        if cell_indices.is_empty() {
            continue;
        }

        let size = cell_indices.len();
        let umis: f64 = cell_indices.iter().map(|&i| umis_per_cell[i] as f64).sum();

        // Rule 1: hard floor.
        if size < min_metacell_size {
            continue;
        }

        // Rule 2: robust size or UMIs.
        if size >= min_robust_size || umis >= min_robust_umis {
            keep[c] = true;
            continue;
        }

        // Rule 3: no convincing-gene test → keep.
        let Some(min_fold) = params.min_convincing_gene_fold_factor else {
            keep[c] = true;
            continue;
        };

        // Rule 4: convincing-gene test. For each gene, compare actual UMIs in
        // this candidate to expected UMIs from the population fraction scaled
        // by the candidate's total. Keep iff any gene has
        // |log2((actual+1)/(expected+1))| >= min_fold.
        let candidate_total: f64 = umis;
        let mut keep_this = false;
        let mut candidate_per_gene = vec![0.0_f64; n_genes];
        for &cell in &cell_indices {
            let start = raw_umis.indptr[cell];
            let end = raw_umis.indptr[cell + 1];
            for idx in start..end {
                let g = raw_umis.indices[idx];
                candidate_per_gene[g] += raw_umis.data[idx] as f64;
            }
        }
        for gene in 0..n_genes {
            let actual = candidate_per_gene[gene] + 1.0;
            let expected = fraction_per_gene[gene] * candidate_total + 1.0;
            let fold = (actual / expected).log2().abs();
            if fold >= min_fold as f64 {
                keep_this = true;
                break;
            }
        }
        keep[c] = keep_this;
    }

    let mut new_id = vec![-1_i32; n_candidates];
    let mut next = 0_i32;
    for c in 0..n_candidates {
        if keep[c] {
            new_id[c] = next;
            next += 1;
        }
    }

    let mut metacell_of_cell = vec![-1_i32; n_cells];
    let mut dissolved_of_cell = vec![false; n_cells];
    for cell in 0..n_cells {
        let c = candidate_of_cell[cell];
        if c < 0 || deviant_of_cell[cell] {
            continue;
        }
        let c = c as usize;
        if keep[c] {
            metacell_of_cell[cell] = new_id[c];
        } else {
            dissolved_of_cell[cell] = true;
        }
    }

    (metacell_of_cell, dissolved_of_cell)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::sparse::CompressedSparseFormat;

    fn make_raw(rows: Vec<Vec<(usize, u32)>>, n_genes: usize) -> CompressedSparseData2<u32, f32> {
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0usize];
        for row in &rows {
            for &(g, u) in row {
                data.push(u);
                indices.push(g);
            }
            indptr.push(data.len());
        }
        CompressedSparseData2 {
            data,
            indices,
            indptr,
            cs_type: CompressedSparseFormat::Csr,
            data_2: None,
            shape: (rows.len(), n_genes),
        }
    }

    #[test]
    fn dissolves_below_min_size() {
        // Two cells in one candidate, min_metacell_size = 3 → dissolve.
        let raw = make_raw(vec![vec![(0, 10)], vec![(0, 10)]], 2);
        let umis = vec![10.0_f32, 10.0];
        let cand = vec![0_i32, 0];
        let dev = vec![false, false];
        let params = DissolveParams::default();
        let (mc, dis) = dissolve_metacells(&raw, &umis, &cand, &dev, 4, 40.0, 3, &params);
        assert_eq!(mc, vec![-1, -1]);
        assert_eq!(dis, vec![true, true]);
    }

    #[test]
    fn keeps_robust_size() {
        // 4 cells, target = 4, min_robust_factor = 0.5 → min_robust_size = 2.
        // Size 4 >= 2, keep.
        let raw = make_raw(
            vec![vec![(0, 10)], vec![(0, 10)], vec![(0, 10)], vec![(0, 10)]],
            2,
        );
        let umis = vec![10.0_f32; 4];
        let cand = vec![0_i32; 4];
        let dev = vec![false; 4];
        let params = DissolveParams::default();
        let (mc, dis) = dissolve_metacells(&raw, &umis, &cand, &dev, 4, 40.0, 2, &params);
        assert_eq!(mc, vec![0, 0, 0, 0]);
        assert!(dis.iter().all(|&v| !v));
    }

    #[test]
    fn deviants_become_outliers_not_dissolved() {
        // 4 cells; cell 3 is deviant. Surviving size 3, robust at 2 →
        // metacell 0 keeps. Cell 3 ends up at -1 but not flagged dissolved.
        let raw = make_raw(
            vec![vec![(0, 10)], vec![(0, 10)], vec![(0, 10)], vec![(0, 10)]],
            2,
        );
        let umis = vec![10.0_f32; 4];
        let cand = vec![0_i32; 4];
        let dev = vec![false, false, false, true];
        let params = DissolveParams::default();
        let (mc, dis) = dissolve_metacells(&raw, &umis, &cand, &dev, 4, 40.0, 2, &params);
        assert_eq!(mc, vec![0, 0, 0, -1]);
        assert!(dis.iter().all(|&v| !v));
    }

    #[test]
    fn dense_metacell_ids_with_gaps() {
        // Three candidates; middle one dissolves. Output IDs should be
        // dense [0, 1].
        let raw = make_raw(
            vec![
                vec![(0, 10)],
                vec![(0, 10)], // candidate 0
                vec![(0, 10)], // candidate 1, will dissolve (size < min)
                vec![(0, 10)],
                vec![(0, 10)], // candidate 2
            ],
            2,
        );
        let umis = vec![10.0_f32; 5];
        let cand = vec![0_i32, 0, 1, 2, 2];
        let dev = vec![false; 5];
        let params = DissolveParams {
            min_convincing_gene_fold_factor: None,
            ..Default::default()
        };
        let (mc, _dis) = dissolve_metacells(&raw, &umis, &cand, &dev, 4, 40.0, 2, &params);
        // Candidate 0 → metacell 0, candidate 1 dissolves, candidate 2 →
        // metacell 1.
        assert_eq!(mc, vec![0, 0, -1, 1, 1]);
    }
}
