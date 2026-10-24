//! Deviant cell detection, upstream's `gaps` policy
//! (`metacells/tools/deviants.py`, `gaps.cpp`).
//!
//! ### Algorithm
//!
//! For each candidate metacell, for each gene:
//!
//! 1. Sort the candidate's non-deviant cells by `log2(fraction + reg)` of
//!    that gene, where `reg = 1 / max(library_size, quantile)` and the
//!    quantile is over the candidate's library sizes.
//! 2. Walk the sorted list from both ends looking for a gap of at least the
//!    gene's minimum fold between positions `p` and `p + skip`. Cells beyond a
//!    gap record its excess as their gap score.
//!
//! The inner loop repeats the pass, marking the cells with the largest gap
//! scores as deviants, until none are new or `max_cell_fraction` is reached;
//! if exceeded, the outer loop raises every threshold by 1/8 and restarts.
//!
//! Each candidate gathers only its members' sparse rows and only the genes
//! where some member clears `min_compare_umis`, so no dense
//! `n_cells × n_genes` matrix is ever built.

use rayon::prelude::*;

use crate::prelude::*;

use super::params::DeviantsParams;

////////////
// Consts //
////////////

/// Upstream's threshold step when too many cells are deviant.
const THRESHOLD_STEP: f32 = 1.0 / 8.0;

/// Smallest candidate (after removing deviants) the gap pass looks at.
const MIN_GAP_CANDIDATE_CELLS: usize = 4;

/// Tolerance upstream adds to the UMI comparisons.
const UMI_TOLERANCE: f32 = 1e-6;

/////////////
// Helpers //
/////////////

/// Group cell indices by candidate id, ascending within each group.
///
/// ### Params
///
/// * `candidate_of_cell` - Candidate assignment per cell; entries outside
///   `0..n_candidates` are skipped.
/// * `n_candidates` - Number of candidates.
///
/// ### Returns
///
/// One cell list per candidate.
pub(super) fn bucket_cells_by_candidate(
    candidate_of_cell: &[i32],
    n_candidates: usize,
) -> Vec<Vec<usize>> {
    let mut buckets = vec![Vec::new(); n_candidates];
    for (cell, &c) in candidate_of_cell.iter().enumerate() {
        if c >= 0 && (c as usize) < n_candidates {
            buckets[c as usize].push(cell);
        }
    }
    buckets
}

/// Linear quantile of `f32` values, computed in `f64` like numpy.
///
/// ### Params
///
/// * `values` - Values, any order.
/// * `q` - Quantile in `[0, 1]`.
///
/// ### Returns
///
/// The quantile, `0` for an empty slice.
fn quantile_f64(values: &[f32], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.iter().map(|&v| v as f64).collect();
    sorted.sort_unstable_by(f64::total_cmp);
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Per-cell regularisation `1 / max(library_size, max(1, quantile))`, the
/// quantile taken over the library sizes of the cell's candidate.
///
/// ### Params
///
/// * `umis_per_cell` - Per-cell library sizes.
/// * `cells_of_candidate` - Cell indices per candidate.
/// * `regularisation_quantile` - Quantile of the candidate's library sizes.
///
/// ### Returns
///
/// The regularisation per cell; `0` for cells without a candidate.
fn compute_regularisation_per_cell(
    umis_per_cell: &[f32],
    cells_of_candidate: &[Vec<usize>],
    regularisation_quantile: f32,
) -> Vec<f32> {
    let mut reg = vec![0.0_f32; umis_per_cell.len()];
    for cells in cells_of_candidate {
        let sizes: Vec<f32> = cells.iter().map(|&i| umis_per_cell[i]).collect();
        let floor = (quantile_f64(&sizes, regularisation_quantile as f64) as f32).max(1.0);
        for &i in cells {
            reg[i] = 1.0 / umis_per_cell[i].max(floor);
        }
    }
    reg
}

/// One candidate's members restricted to the genes worth testing, gene-major.
struct CandidateSlab {
    /// Number of member cells (columns of each gene row).
    n_members: usize,
    /// Genes in the slab, ascending.
    genes: Vec<usize>,
    /// UMIs, `genes.len() × n_members`, gene-major.
    umis: Vec<f32>,
    /// `log2(umis / library + reg)`, same layout.
    log_fractions: Vec<f32>,
}

/// Gather a candidate's members into a [`CandidateSlab`].
///
/// Keeps a gene when some member satisfies upstream's
/// `umis * 2 + 1e-6 > min_compare_umis`; other genes cannot produce a gap.
/// Genes without any positive count in the candidate are always left out,
/// which only differs from upstream for `min_compare_umis <= 1e-6`. Cells
/// absent from a gene's sparse row have zero UMIs and log fraction
/// `log2(reg)`.
///
/// ### Params
///
/// * `raw_umis` - Cell-major raw counts.
/// * `members` - Member cells in order.
/// * `umis_per_cell` - Library sizes.
/// * `reg_per_cell` - Regularisation per cell.
/// * `min_compare_umis` - Upstream's `min_compare_umis`.
///
/// ### Returns
///
/// The slab.
fn gather_candidate(
    raw_umis: &CompressedSparseData2<u32, f32>,
    members: &[usize],
    umis_per_cell: &[f32],
    reg_per_cell: &[f32],
    min_compare_umis: f32,
) -> CandidateSlab {
    let n_members = members.len();

    let mut entries: Vec<(u32, u32, u32)> = Vec::new();
    for (pos, &cell) in members.iter().enumerate() {
        let start = raw_umis.indptr[cell] as usize;
        let end = raw_umis.indptr[cell + 1] as usize;
        for idx in start..end {
            if raw_umis.data[idx] > 0 {
                entries.push((raw_umis.indices[idx], pos as u32, raw_umis.data[idx]));
            }
        }
    }
    entries.sort_unstable_by_key(|&(g, pos, _)| (g, pos));

    let base_lf: Vec<f32> = members.iter().map(|&c| reg_per_cell[c].log2()).collect();

    let mut genes = Vec::new();
    let mut umis = Vec::new();
    let mut log_fractions = Vec::new();
    for group in entries.chunk_by(|a, b| a.0 == b.0) {
        let significant = group
            .iter()
            .any(|&(_, _, u)| (u as f32) * 2.0 + UMI_TOLERANCE > min_compare_umis);
        if !significant {
            continue;
        }
        genes.push(group[0].0 as usize);
        let row_start = umis.len();
        umis.resize(row_start + n_members, 0.0);
        log_fractions.extend_from_slice(&base_lf);
        for &(_, pos, u) in group {
            let cell = members[pos as usize];
            let lib = umis_per_cell[cell];
            // upstream divides; a zero library keeps the zero fraction
            let fraction = if lib > 0.0 { u as f32 / lib } else { 0.0 };
            umis[row_start + pos as usize] = u as f32;
            log_fractions[row_start + pos as usize] = (fraction + reg_per_cell[cell]).log2();
        }
    }

    CandidateSlab {
        n_members,
        genes,
        umis,
        log_fractions,
    }
}

/// Record the per-cell maximum gap excess within one candidate, as upstream's
/// `compute_cell_gaps` for one candidate.
///
/// ### Params
///
/// * `slab` - The candidate's gathered data.
/// * `members` - Member cells, aligned with the slab columns.
/// * `deviant_per_cell` - Cells already deviant; left out of the sort.
/// * `min_gap_per_gene` - Minimum fold per gene.
/// * `gap_skip_cells` - Internal skip, in `1..=3`.
/// * `params` - Deviant parameters.
/// * `max_gap` - Maximum gap per member position; updated in place.
fn candidate_gaps(
    slab: &CandidateSlab,
    deviant_per_cell: &[bool],
    members: &[usize],
    min_gap_per_gene: &[f32],
    gap_skip_cells: usize,
    params: &DeviantsParams,
    max_gap: &mut [f32],
) {
    let cells: Vec<usize> = (0..slab.n_members)
        .filter(|&pos| !deviant_per_cell[members[pos]])
        .collect();
    let n = cells.len();
    if n < MIN_GAP_CANDIDATE_CELLS {
        return;
    }

    let max_deviant = (n - gap_skip_cells).min(
        params
            .max_gap_cells_count
            .max((params.max_gap_cells_fraction * n as f32 + 0.5) as usize),
    );
    let max_cell_position = ((n - 1) / 2).min(max_deviant);
    let min_compare = params.min_compare_umis;

    let mut umi_gene = vec![0.0_f32; n];
    let mut lf_gene = vec![0.0_f32; n];
    let mut sorted: Vec<usize> = Vec::with_capacity(n);

    for (row, &gene) in slab.genes.iter().enumerate() {
        let offset = row * slab.n_members;
        for (slot, &pos) in cells.iter().enumerate() {
            umi_gene[slot] = slab.umis[offset + pos];
            lf_gene[slot] = slab.log_fractions[offset + pos];
        }
        if !umi_gene
            .iter()
            .any(|&u| u * 2.0 + UMI_TOLERANCE > min_compare)
        {
            continue;
        }
        let min_gap = min_gap_per_gene[gene];

        sorted.clear();
        sorted.extend(0..n);
        sorted.sort_unstable_by(|&a, &b| lf_gene[a].total_cmp(&lf_gene[b]));

        // low end: upward gaps
        for cell_pos in 0..max_cell_position {
            let first = sorted[cell_pos];
            let second = sorted[cell_pos + gap_skip_cells];
            if umi_gene[first] + umi_gene[second] + UMI_TOLERANCE < min_compare {
                continue;
            }
            let full_gap = lf_gene[second] - lf_gene[first] - min_gap;
            if full_gap < 0.0 {
                continue;
            }
            let mut stop = cell_pos + 1;
            if gap_skip_cells > 1 {
                let middle = sorted[cell_pos + 1];
                let start_gap = 2.0 * (lf_gene[middle] - lf_gene[first]) - min_gap;
                if start_gap < full_gap {
                    stop += 1;
                }
            }
            for &slot in &sorted[..=stop] {
                let g = &mut max_gap[cells[slot]];
                *g = g.max(full_gap);
            }
        }

        // high end: downward gaps; upstream drops the tolerance here
        for cell_offset in 0..max_cell_position {
            let cell_pos = n - 1 - cell_offset - gap_skip_cells;
            let first = sorted[cell_pos];
            let second = sorted[cell_pos + gap_skip_cells];
            if umi_gene[first] + umi_gene[second] < min_compare {
                continue;
            }
            let full_gap = lf_gene[second] - lf_gene[first] - min_gap;
            if full_gap <= 0.0 {
                continue;
            }
            let mut start = cell_pos + 1;
            if gap_skip_cells > 1 {
                let middle = sorted[cell_pos + 1];
                let start_gap = 2.0 * (lf_gene[middle] - lf_gene[first]) - min_gap;
                if start_gap < full_gap {
                    start += 1;
                }
            }
            for &slot in &sorted[start..] {
                let g = &mut max_gap[cells[slot]];
                *g = g.max(full_gap);
            }
        }
    }
}

//////////
// Main //
//////////

/// Top-level deviant detection.
///
/// ### Params
///
/// * `raw_umis` - `(n_cells, n_genes)` raw UMI counts in CSR.
/// * `umis_per_cell` - Row sums of `raw_umis`.
/// * `candidate_of_cell` - Candidate metacell per cell; `-1` cells are
///   skipped and never flagged.
/// * `params` - Deviant detection parameters.
///
/// ### Returns
///
/// A boolean mask where `true` means deviant.
///
/// ### Errors
///
/// * `Mc2InvalidParams` if `max_cell_fraction` is outside `(0, 1)` or
///   `gap_skip_cells > 2`.
/// * `Mc2LengthMismatch` if the noisy gene mask does not cover every gene.
pub fn find_deviant_cells(
    raw_umis: &CompressedSparseData2<u32, f32>,
    umis_per_cell: &[f32],
    candidate_of_cell: &[i32],
    params: &DeviantsParams,
) -> Result<Vec<bool>, ScErrors> {
    let n_cells = raw_umis.shape.0;
    let n_genes = raw_umis.shape.1;
    assert_eq!(umis_per_cell.len(), n_cells);
    assert_eq!(candidate_of_cell.len(), n_cells);

    if !(params.max_cell_fraction > 0.0 && params.max_cell_fraction < 1.0) {
        return Err(ScErrors::Mc2InvalidParams {
            reason: format!(
                "deviants max_cell_fraction must be in (0, 1), got {}",
                params.max_cell_fraction
            ),
        });
    }
    if params.gap_skip_cells > 2 {
        return Err(ScErrors::Mc2InvalidParams {
            reason: format!(
                "deviants gap_skip_cells must be in 0..=2, got {}",
                params.gap_skip_cells
            ),
        });
    }
    if let Some(mask) = params.noisy_gene_mask.as_deref()
        && mask.len() != n_genes
    {
        return Err(ScErrors::Mc2LengthMismatch {
            what: "noisy_gene_mask",
            expected: n_genes,
            got: mask.len(),
        });
    }

    if n_cells <= params.max_gap_cells_count {
        return Ok(vec![false; n_cells]);
    }
    let n_candidates = (candidate_of_cell.iter().copied().max().unwrap_or(-1) + 1).max(0) as usize;
    if n_candidates == 0 {
        return Ok(vec![false; n_cells]);
    }

    let gap_skip_cells = params.gap_skip_cells + 1;
    let max_cell_fraction = params.max_cell_fraction as f64;

    let mut min_gap_per_gene = vec![params.min_gene_fold_factor; n_genes];
    if let Some(mask) = params.noisy_gene_mask.as_deref() {
        for (g, &noisy) in min_gap_per_gene.iter_mut().zip(mask) {
            if noisy {
                *g += params.min_noisy_gene_fold_factor;
            }
        }
    }

    let cells_of_candidate = bucket_cells_by_candidate(candidate_of_cell, n_candidates);
    let reg_per_cell = compute_regularisation_per_cell(
        umis_per_cell,
        &cells_of_candidate,
        params.cells_regularisation_quantile,
    );
    let slabs: Vec<CandidateSlab> = cells_of_candidate
        .par_iter()
        .map(|members| {
            gather_candidate(
                raw_umis,
                members,
                umis_per_cell,
                &reg_per_cell,
                params.min_compare_umis,
            )
        })
        .collect();

    loop {
        let mut deviant_per_cell = vec![false; n_cells];
        let mut active_per_cell = vec![true; n_cells];
        let mut deviants_fraction = 0.0_f64;

        while deviants_fraction <= max_cell_fraction {
            // per candidate gaps, written back into the per-cell vector
            let per_candidate: Vec<Vec<f32>> = slabs
                .par_iter()
                .zip(cells_of_candidate.par_iter())
                .map(|(slab, members)| {
                    let mut gaps = vec![0.0_f32; members.len()];
                    if members.iter().any(|&c| active_per_cell[c]) {
                        candidate_gaps(
                            slab,
                            &deviant_per_cell,
                            members,
                            &min_gap_per_gene,
                            gap_skip_cells,
                            params,
                            &mut gaps,
                        );
                    }
                    gaps
                })
                .collect();
            let mut max_gap_per_cell = vec![0.0_f32; n_cells];
            for (gaps, members) in per_candidate.iter().zip(&cells_of_candidate) {
                for (&g, &c) in gaps.iter().zip(members) {
                    max_gap_per_cell[c] = g;
                }
            }

            let large_gap_count = max_gap_per_cell.iter().filter(|&&v| v > 0.0).count();
            let remaining_fraction = 1.0 - (max_cell_fraction - deviants_fraction).max(0.0);
            let gap_threshold = if (large_gap_count as f64) / (n_cells as f64) <= remaining_fraction
            {
                0.0_f32
            } else {
                quantile_f64(&max_gap_per_cell, remaining_fraction) as f32
            };

            active_per_cell = max_gap_per_cell
                .iter()
                .zip(&deviant_per_cell)
                .map(|(&g, &d)| g > gap_threshold && !d)
                .collect();
            let new_count = active_per_cell.iter().filter(|&&v| v).count();
            if new_count == 0 {
                break;
            }
            deviants_fraction += new_count as f64 / n_cells as f64;
            for (d, &nd) in deviant_per_cell.iter_mut().zip(&active_per_cell) {
                *d |= nd;
            }
        }

        if deviants_fraction <= max_cell_fraction {
            return Ok(deviant_per_cell);
        }

        for g in min_gap_per_gene.iter_mut() {
            *g += THRESHOLD_STEP;
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn make_csr(rows: Vec<Vec<(usize, u32)>>, n_genes: usize) -> CompressedSparseData2<u32, f32> {
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
            indices: indices.index_cast(),
            indptr: indptr.index_cast(),
            cs_type: CompressedSparseFormat::Csr,
            data_2: None,
            shape: (rows.len(), n_genes),
        }
    }

    /// Candidates too small to compare against are left free of deviants.
    #[test]
    fn test_deviants_none_when_candidate_too_small() {
        let raw = make_csr(
            vec![
                vec![(0, 100), (1, 1)],
                vec![(0, 99), (1, 2)],
                vec![(0, 101), (1, 1)],
                vec![(0, 100), (1, 1)],
            ],
            2,
        );
        let umis = vec![101.0_f32, 101.0, 102.0, 101.0];
        let params = DeviantsParams {
            max_gap_cells_count: 1,
            ..Default::default()
        };
        // two candidates of two cells, each below the four-cell floor
        let dev = find_deviant_cells(&raw, &umis, &[0, 0, 1, 1], &params).unwrap();
        assert_eq!(dev, vec![false; 4]);
    }

    /// A single cell with a large fold change on one gene is flagged.
    #[test]
    fn test_deviants_detects_clear_outlier() {
        let raw = make_csr(
            vec![
                vec![(0, 100), (1, 50)],
                vec![(0, 102), (1, 50)],
                vec![(0, 98), (1, 50)],
                vec![(0, 101), (1, 50)],
                vec![(0, 1), (1, 50)],
                vec![(0, 99), (1, 50)],
                vec![(0, 103), (1, 50)],
                vec![(0, 97), (1, 50)],
            ],
            2,
        );
        let umis: Vec<f32> = vec![150.0, 152.0, 148.0, 151.0, 51.0, 149.0, 153.0, 147.0];
        let params = DeviantsParams {
            min_gene_fold_factor: 1.0,
            min_compare_umis: 5.0,
            gap_skip_cells: 0,
            max_cell_fraction: 0.5,
            max_gap_cells_count: 2,
            max_gap_cells_fraction: 0.4,
            cells_regularisation_quantile: 0.1,
            ..Default::default()
        };
        let dev = find_deviant_cells(&raw, &umis, &[0; 8], &params).unwrap();
        assert!(dev[4], "expected cell 4 to be flagged: {dev:?}");
        let clean = [0_usize, 1, 2, 3, 5, 6, 7]
            .iter()
            .filter(|&&i| !dev[i])
            .count();
        assert!(
            clean >= 5,
            "expected >= 5 clean cells, got {clean}: {dev:?}"
        );
    }

    /// Out-of-range parameters are rejected rather than clamped.
    #[test]
    fn test_deviants_rejects_bad_params() {
        let raw = make_csr(vec![vec![(0, 1)]; 5], 1);
        let umis = vec![1.0; 5];
        let bad_skip = DeviantsParams {
            gap_skip_cells: 3,
            ..Default::default()
        };
        assert!(find_deviant_cells(&raw, &umis, &[0; 5], &bad_skip).is_err());
        let bad_fraction = DeviantsParams {
            max_cell_fraction: 1.0,
            ..Default::default()
        };
        assert!(find_deviant_cells(&raw, &umis, &[0; 5], &bad_fraction).is_err());
    }
}
