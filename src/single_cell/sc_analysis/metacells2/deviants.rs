//! Deviant cell detection ("gaps" policy).
//!
//! ### Algorithm
//!
//! For each candidate metacell, for each gene:
//!
//! 1. Sort the metacell's non-deviant cells by `log2(fraction + reg)` of
//!    that gene, where `reg` is `1 / max(library_size,
//!    quantile(library_sizes, q))` (per-cell, but constant across genes
//!    within the cell).
//! 2. Walk the sorted list from both ends. Look for a gap of at least
//!    `min_gene_fold_factor` between the cell at position `p` and the cell
//!    at position `p + skip`. When found, mark cells `[0..=p+stop]` (low
//!    end) or `[p+start..end]` (high end) as having a deviant gap.
//! 3. The skip mechanic absorbs local noise: a small skip might bridge
//!    over a single noisy cell.
//!
//! Two loops: inner loop iterates the gap pass, marking new deviants each
//! time, until no new deviants emerge. Outer loop bumps the threshold by
//! `1/8` if the deviant fraction exceeds `max_cell_fraction`.

use rayon::prelude::*;

use crate::core::math::vector_helpers::{quantile, quantile_sorted};
use crate::prelude::*;

use super::params::DeviantsParams;

/////////////
// Helpers //
/////////////

/// Build dense row-major `(n_cells × n_genes)` UMI and fraction matrices.
///
/// Both matrices are `Vec<f32>` of length `n_cells * n_genes`, indexed by
/// `cell * n_genes + gene`. Fraction is `umi / library_size` for non-zero
/// library sizes, and zero otherwise.
///
/// ### Params
///
/// * `raw_umis`      - CSR matrix of raw UMI counts.
/// * `umis_per_cell` - Pre-summed library size per cell.
///
/// ### Returns
///
/// A tuple `(umi_dense, fraction_dense)` in row-major layout.
fn build_dense_views(
    raw_umis: &CompressedSparseData2<u32, f32>,
    umis_per_cell: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let n_cells = raw_umis.shape.0;
    let n_genes = raw_umis.shape.1;
    let total = n_cells * n_genes;
    let mut umi = vec![0.0_f32; total];
    let mut frac = vec![0.0_f32; total];

    for cell in 0..n_cells {
        let lib = umis_per_cell[cell];
        let inv_lib = if lib > 0.0 { 1.0 / lib } else { 0.0 };
        let row_base = cell * n_genes;
        let start = raw_umis.indptr[cell];
        let end = raw_umis.indptr[cell + 1];
        for idx in start..end {
            let g = raw_umis.indices[idx];
            let u = raw_umis.data[idx] as f32;
            umi[row_base + g] = u;
            frac[row_base + g] = u * inv_lib;
        }
    }

    (umi, frac)
}

/// Compute per-cell regularisation values as
/// `1 / max(library_size, quantile_floor)`.
///
/// The quantile floor is computed separately for each candidate metacell from
/// the distribution of library sizes of its members, so cells in low-quality
/// candidates get a tighter floor.
///
/// ### Params
///
/// * `umis_per_cell` - Per-cell library sizes.
/// * `candidate_of_cell` - Candidate assignment per cell; `< 0` entries are
///   skipped.
/// * `n_candidates` - Total number of candidates.
/// * `regularisation_quantile`   - Quantile of the candidate's library sizes
///   used as the floor.
///
/// ### Returns
///
/// A `Vec<f32>` of length `n_cells` with the regularisation value per cell.
/// Cells without a valid candidate retain `0.0` and are skipped downstream.
fn compute_regularisation_per_cell(
    umis_per_cell: &[f32],
    candidate_of_cell: &[i32],
    n_candidates: usize,
    regularisation_quantile: f32,
) -> Vec<f32> {
    let n_cells = umis_per_cell.len();
    let mut reg = vec![0.0_f32; n_cells];

    // per-candidate quantile of library sizes.
    for c in 0..n_candidates {
        let mut sizes: Vec<f32> = umis_per_cell
            .iter()
            .zip(candidate_of_cell.iter())
            .filter_map(|(&u, &cand)| if cand == c as i32 { Some(u) } else { None })
            .collect();
        if sizes.is_empty() {
            continue;
        }
        sizes.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q = quantile_sorted(&sizes, regularisation_quantile);
        let floor = q.max(1.0);

        for (i, &cand) in candidate_of_cell.iter().enumerate() {
            if cand == c as i32 {
                let denom = umis_per_cell[i].max(floor);
                reg[i] = 1.0 / denom;
            }
        }
    }

    reg
}

/// Compute `log2(fraction + reg)` for every `(cell, gene)` pair.
///
/// Values where `fraction + reg <= 0` are set to `NEG_INFINITY`. Output is
/// row-major with the same layout as the input fraction matrix.
///
/// ### Params
///
/// * `fraction_dense` - Row-major fraction matrix of shape `(n_cells, n_genes)`.
/// * `reg_per_cell` - Per-cell regularisation values.
/// * `n_cells` - Number of cells.
/// * `n_genes` - Number of genes.
///
/// ### Returns
///
/// A `Vec<f32>` of length `n_cells * n_genes` in row-major layout.
fn compute_log_fractions(
    fraction_dense: &[f32],
    reg_per_cell: &[f32],
    n_cells: usize,
    n_genes: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; n_cells * n_genes];
    out.par_chunks_mut(n_genes)
        .zip(fraction_dense.par_chunks(n_genes))
        .zip(reg_per_cell.par_iter())
        .for_each(|((out_row, frac_row), &reg)| {
            for (o, &f) in out_row.iter_mut().zip(frac_row.iter()) {
                let v = f + reg;
                *o = if v > 0.0 { v.log2() } else { f32::NEG_INFINITY };
            }
        });
    out
}

/// Compute the per-cell maximum gap score across all genes and candidates.
///
/// For each candidate and each gene, sorts the candidate's non-deviant cells
/// by their log-fraction, scans from both ends for gaps exceeding
/// `min_gap_per_gene`, and records the gap magnitude in `max_gap_per_cell`
/// for each cell in the deviant range. Runs candidates in parallel; writes
/// are safe because each cell belongs to exactly one candidate.
///
/// ### Params
///
/// * `umi_dense` - Row-major UMI matrix `(n_cells × n_genes)`.
/// * `log_fraction_dense` - Row-major log-fraction matrix
///   `(n_cells × n_genes)`.
/// * `candidate_of_cell` - Candidate assignment per cell.
/// * `deviant_per_cell` - Cells already marked deviant; excluded from the
///   sorted list.
/// * `active_per_cell` - Cells active in the current inner-loop iteration.
/// * `min_gap_per_gene` - Per-gene minimum gap threshold.
/// * `n_cells` - Number of cells.
/// * `n_genes` - Number of genes.
/// * `n_candidates` - Number of candidate metacells.
/// * `gap_skip_cells` - Number of positions to skip when measuring a gap; in
///   `[1, 3]`.
/// * `max_gap_cells_count` - Absolute maximum number of deviant cells per
///   candidate.
/// * `max_gap_cells_fraction`- Fractional maximum number of deviant cells per
///   candidate.
/// * `min_compare_umis` - Minimum combined UMI count for a gene to be tested.
/// * `max_gap_per_cell` - Output buffer of length `n_cells`; updated in place.
///
/// ### Returns
///
/// Nothing; `max_gap_per_cell` is populated in place.
#[allow(clippy::too_many_arguments)]
fn compute_cell_gaps(
    umi_dense: &[f32],
    log_fraction_dense: &[f32],
    candidate_of_cell: &[i32],
    deviant_per_cell: &[bool],
    active_per_cell: &[bool],
    min_gap_per_gene: &[f32],
    n_cells: usize,
    n_genes: usize,
    n_candidates: usize,
    gap_skip_cells: usize,
    max_gap_cells_count: usize,
    max_gap_cells_fraction: f32,
    min_compare_umis: f32,
    max_gap_per_cell: &mut [f32],
) {
    // deal with rayon limitations here with some raw pointer magic
    let max_gap_addr = max_gap_per_cell.as_mut_ptr() as usize;

    (0..n_candidates).into_par_iter().for_each(|cand| {
        // collect non-deviant cells in this candidate, plus check if any
        // active cells exist (carry-forward from outer loop convergence).
        let mut cells: Vec<usize> = Vec::with_capacity(n_cells / n_candidates.max(1));
        let mut active_candidate = false;
        for cell in 0..n_cells {
            if candidate_of_cell[cell] != cand as i32 {
                continue;
            }
            if active_per_cell[cell] {
                active_candidate = true;
            }
            if !deviant_per_cell[cell] {
                cells.push(cell);
            }
        }
        let candidate_cells_count = cells.len();
        if !active_candidate || candidate_cells_count < 4 {
            return;
        }

        let max_deviant_of_candidate = (candidate_cells_count - gap_skip_cells).min(
            max_gap_cells_count
                .max((max_gap_cells_fraction * candidate_cells_count as f32 + 0.5) as usize),
        );

        // SAFETY: candidates partition the cells; writes through max_gap are to
        // disjoint subsets across rayon tasks.
        let max_gap = unsafe { std::slice::from_raw_parts_mut(max_gap_addr as *mut f32, n_cells) };

        // Working buffer for sorted cell positions per gene. Reused across
        // genes in this candidate.
        let mut sorted = cells.clone();

        for gene in 0..n_genes {
            let min_gap = min_gap_per_gene[gene];

            // Skip if no cell in this candidate has enough UMIs to be
            // considered "reliably expressed" for this gene.
            let has_significant = cells.iter().any(|&c| {
                let u = umi_dense[c * n_genes + gene];
                u * 2.0 + 1e-6 > min_compare_umis
            });
            if !has_significant {
                continue;
            }

            // Reset and sort by log_fraction[gene].
            sorted.copy_from_slice(&cells);
            sorted.sort_unstable_by(|&a, &b| {
                let la = log_fraction_dense[a * n_genes + gene];
                let lb = log_fraction_dense[b * n_genes + gene];
                la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
            });

            let max_cell_position = ((candidate_cells_count - 1) / 2).min(max_deviant_of_candidate);

            // -- Low end: scan from bottom for upward gaps.
            for cell_pos in 0..max_cell_position {
                let first = sorted[cell_pos];
                let second = sorted[cell_pos + gap_skip_cells];

                let first_u = umi_dense[first * n_genes + gene];
                let second_u = umi_dense[second * n_genes + gene];
                if first_u + second_u + 1e-6 < min_compare_umis {
                    continue;
                }

                let first_lf = log_fraction_dense[first * n_genes + gene];
                let second_lf = log_fraction_dense[second * n_genes + gene];
                let full_gap = second_lf - first_lf - min_gap;
                if full_gap < 0.0 {
                    continue;
                }

                // Look-ahead: when skip > 1, check the half-gap; if it's
                // smaller than the full gap, extend the deviant range by 1.
                let mut stop_cell_position = cell_pos + 1;
                if gap_skip_cells > 1 {
                    let middle = sorted[cell_pos + 1];
                    let middle_lf = log_fraction_dense[middle * n_genes + gene];
                    let start_gap = 2.0 * (middle_lf - first_lf) - min_gap;
                    if start_gap < full_gap {
                        stop_cell_position += 1;
                    }
                }

                for gap_cell_position in 0..=stop_cell_position {
                    let gap_cell = sorted[gap_cell_position];
                    let cur = max_gap[gap_cell];
                    if full_gap > cur {
                        max_gap[gap_cell] = full_gap;
                    }
                }
            }

            // -- High end: scan from top for downward gaps. Mirror of above.
            for cell_offset in 0..max_cell_position {
                let cell_pos = candidate_cells_count - 1 - cell_offset - gap_skip_cells;
                let first = sorted[cell_pos];
                let second = sorted[cell_pos + gap_skip_cells];

                let first_u = umi_dense[first * n_genes + gene];
                let second_u = umi_dense[second * n_genes + gene];
                if first_u + second_u < min_compare_umis {
                    continue;
                }

                let first_lf = log_fraction_dense[first * n_genes + gene];
                let second_lf = log_fraction_dense[second * n_genes + gene];
                let full_gap = second_lf - first_lf - min_gap;
                if full_gap <= 0.0 {
                    continue;
                }

                let mut start_cell_position = cell_pos + 1;
                if gap_skip_cells > 1 {
                    let middle = sorted[cell_pos + 1];
                    let middle_lf = log_fraction_dense[middle * n_genes + gene];
                    let start_gap = 2.0 * (middle_lf - first_lf) - min_gap;
                    if start_gap < full_gap {
                        start_cell_position += 1;
                    }
                }

                for gap_cell_position in start_cell_position..candidate_cells_count {
                    let gap_cell = sorted[gap_cell_position];
                    let cur = max_gap[gap_cell];
                    if full_gap > cur {
                        max_gap[gap_cell] = full_gap;
                    }
                }
            }
        }
    });
}

//////////
// Main //
//////////

/// Top-level deviant detection.
///
/// ### Params
///
/// * `raw_umis` - `(n_cells, n_genes)` raw UMI counts in CSR.
/// * `umis_per_cell` - Pre-summed library size per cell. Used for the
///   per-cell regularisation quantile.
/// * `candidate_of_cell` - Cell-level candidate metacell assignment;
///   `-1` for already-outlier cells (these are skipped).
/// * `params` - Deviant detection parameters.
///
/// ### Returns
///
/// A boolean mask where `true` means "deviant; should not be in any
/// metacell". Cells with `candidate_of_cell == -1` always return `false`
/// from this function (they're already outliers; the caller maintains the
/// outlier flag separately).
pub fn find_deviant_cells(
    raw_umis: &CompressedSparseData2<u32, f32>,
    umis_per_cell: &[f32],
    candidate_of_cell: &[i32],
    params: &DeviantsParams,
) -> Vec<bool> {
    let n_cells = raw_umis.shape.0;
    let n_genes = raw_umis.shape.1;
    assert_eq!(umis_per_cell.len(), n_cells);
    assert_eq!(candidate_of_cell.len(), n_cells);

    if n_cells <= params.max_gap_cells_count {
        return vec![false; n_cells];
    }

    let n_candidates = (candidate_of_cell.iter().copied().max().unwrap_or(-1) + 1) as usize;
    if n_candidates == 0 {
        return vec![false; n_cells];
    }

    // build dense (n_cells × n_genes) raw UMI and fraction matrices.
    let (umi_dense, fraction_dense) = build_dense_views(raw_umis, umis_per_cell);

    // per-cell regularisation: max(library_size, quantile of candidate's
    // library sizes) -> stored as 1/value so it adds directly to the fraction
    // before the log
    let regularisation_per_cell = compute_regularisation_per_cell(
        umis_per_cell,
        candidate_of_cell,
        n_candidates,
        params.cells_regularization_quantile,
    );

    // log2(fraction + reg) per (cell, gene). reg is per-cell, broadcast.
    let log_fraction_dense =
        compute_log_fractions(&fraction_dense, &regularisation_per_cell, n_cells, n_genes);

    // per-gene minimum gap.
    let min_gap_per_gene = vec![params.min_gene_fold_factor; n_genes];

    let max_cell_fraction = params.max_cell_fraction.clamp(1e-6, 1.0);
    let mut min_gene_fold_factor = params.min_gene_fold_factor;
    let mut min_gap_per_gene = min_gap_per_gene;

    // gap_skip_cells in the C++ is 1, 2, or 3 (Python adds 1 to the
    // user-facing 0/1/2 value). We use the post-Python value directly.
    let gap_skip_cells = params.gap_skip_cells.clamp(1, 3);

    let mut deviant_per_cell = vec![false; n_cells];

    // outer loop: bump threshold if deviant fraction is too high.
    loop {
        let mut new_deviant_per_cell = vec![true; n_cells];
        deviant_per_cell.iter_mut().for_each(|d| *d = false);
        let mut deviants_fraction = 0.0_f64;

        // inner loop: keep finding new deviants until no more emerge.
        while deviants_fraction <= max_cell_fraction as f64 {
            let mut max_gap_per_cell = vec![0.0_f32; n_cells];

            compute_cell_gaps(
                &umi_dense,
                &log_fraction_dense,
                candidate_of_cell,
                &deviant_per_cell,
                &new_deviant_per_cell,
                &min_gap_per_gene,
                n_cells,
                n_genes,
                n_candidates,
                gap_skip_cells,
                params.max_gap_cells_count,
                params.max_gap_cells_fraction,
                params.min_compare_umis,
                &mut max_gap_per_cell,
            );

            // how many cells have a non-zero gap?
            let large_gap_count = max_gap_per_cell.iter().filter(|&&v| v > 0.0).count();
            let remaining_fraction = 1.0 - (max_cell_fraction as f64 - deviants_fraction).max(0.0);

            let gap_threshold = if (large_gap_count as f64) / (n_cells as f64) <= remaining_fraction
            {
                0.0_f32
            } else {
                quantile(&max_gap_per_cell, remaining_fraction as f32)
            };

            // new deviants: above threshold, not already deviant.
            new_deviant_per_cell = max_gap_per_cell
                .iter()
                .zip(deviant_per_cell.iter())
                .map(|(&g, &d)| g > gap_threshold && !d)
                .collect();

            let new_count = new_deviant_per_cell.iter().filter(|&&v| v).count();
            if new_count == 0 {
                break;
            }
            deviants_fraction += new_count as f64 / n_cells as f64;
            for (d, &nd) in deviant_per_cell.iter_mut().zip(new_deviant_per_cell.iter()) {
                *d |= nd;
            }
        }

        if deviants_fraction <= max_cell_fraction as f64 {
            return deviant_per_cell;
        }

        // too many deviants: bump threshold and retry.
        for g in min_gap_per_gene.iter_mut() {
            *g += 1.0 / 8.0;
        }
        min_gene_fold_factor += 1.0 / 8.0;
        let _ = min_gene_fold_factor; // recorded for diagnostics
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::sparse::CompressedSparseFormat;

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
            indices,
            indptr,
            cs_type: CompressedSparseFormat::Csr,
            data_2: None,
            shape: (rows.len(), n_genes),
        }
    }

    #[test]
    fn quantile_known_values() {
        let v = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile(&v, 0.0) - 1.0).abs() < 1e-6);
        assert!((quantile(&v, 1.0) - 5.0).abs() < 1e-6);
        assert!((quantile(&v, 0.5) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn no_deviants_when_metacell_too_small() {
        // 3 cells, 2 genes; max_gap_cells_count = 1. n_cells = 3 is more
        // than max_gap_cells_count = 1 so it goes through, but the
        // candidate_cells_count check < 4 will reject.
        let raw = make_csr(
            vec![
                vec![(0, 100), (1, 1)],
                vec![(0, 99), (1, 2)],
                vec![(0, 101), (1, 1)],
            ],
            2,
        );
        let umis = vec![101.0_f32, 101.0, 102.0];
        let cand = vec![0_i32, 0, 0];
        let params = DeviantsParams::default();
        let dev = find_deviant_cells(&raw, &umis, &cand, &params);
        assert_eq!(dev, vec![false, false, false]);
    }

    #[test]
    fn detects_clear_outlier_in_metacell() {
        // 8 cells in one candidate. Cells 0..7 except cell 4 express gene
        // 0 strongly; cell 4 doesn't. Gene 1 is uniform across all cells.
        //
        // We expect cell 4 to be detected as a deviant. Note that the C++
        // gap kernel propagates a gap across `gap_skip_cells + 1` adjacent
        // sorted positions, so with `gap_skip_cells = 1` the gap detected
        // at position 0 marks both position 0 and position 1 with the same
        // gap value. The test therefore checks that the obvious outlier is
        // flagged and that the bulk of well-behaved cells is not all
        // flagged — not that exactly one cell is flagged.
        let raw = make_csr(
            vec![
                vec![(0, 100), (1, 50)],
                vec![(0, 102), (1, 50)],
                vec![(0, 98), (1, 50)],
                vec![(0, 101), (1, 50)],
                vec![(0, 1), (1, 50)], // outlier on gene 0
                vec![(0, 99), (1, 50)],
                vec![(0, 103), (1, 50)],
                vec![(0, 97), (1, 50)],
            ],
            2,
        );
        let umis: Vec<f32> = vec![150.0, 152.0, 148.0, 151.0, 51.0, 149.0, 153.0, 147.0];
        let cand = vec![0_i32; 8];
        let params = DeviantsParams {
            min_gene_fold_factor: 1.0, // log2 fold of 2x
            min_compare_umis: 5.0,
            gap_skip_cells: 1,
            max_cell_fraction: 0.5,
            max_gap_cells_count: 2,
            max_gap_cells_fraction: 0.4,
            cells_regularization_quantile: 0.1,
            ..Default::default()
        };
        let dev = find_deviant_cells(&raw, &umis, &cand, &params);

        // The outlier on cell 4 must be detected.
        assert!(dev[4], "expected cell 4 to be flagged: {:?}", dev);

        // At least 5 of the 7 well-behaved cells must not be flagged
        // (one or two adjacent cells in the sorted order may be caught up
        // in the gap propagation, but the bulk should be clean).
        let well_behaved = [0_usize, 1, 2, 3, 5, 6, 7];
        let n_clean = well_behaved.iter().filter(|&&i| !dev[i]).count();
        assert!(
            n_clean >= 5,
            "expected >= 5 clean well-behaved cells, got {}: {:?}",
            n_clean,
            dev
        );
    }
}
