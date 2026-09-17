//! The residual interface shared by scTransform and the analytic Pearson
//! residuals.
//!
//! Both methods answer the same question for a gene: what does this count look
//! like against a null model that only knows about sequencing depth? They
//! differ entirely in how the null is fitted, and not at all in what is done
//! with the answer. Residual variance ranks genes for HVG selection, the
//! residuals themselves go into PCA, and neither is ever stored: the row is
//! dense even where the counts are not, because a zero count still has residual
//! `-mu / sqrt(var)`.
//!
//! So the whole interface is [`ResidualSource::residual_row`], and everything
//! downstream is written once against it.
//!
//! ### Grouping
//!
//! A source carries one model per group, plus a group id per selected cell.
//! That is how a multi-sample experiment is handled: each sample is fitted
//! independently and every cell is scored under its own sample's model, with
//! the gene axis narrowed to what every sample modelled. A single-sample fit is
//! the same code with one group.
//!
//! The cell axis matters. `read_gene_parallel_filtered` reindexes each gene's
//! non-zeros to positions inside the selected cell set, in `cell_indices`
//! order, so [`ResidualSource::group_of_cell`], the `indices` handed to
//! `residual_row` and its `out` row all agree. Do not reach for
//! [`CellBatchIndex`](super::hvg::CellBatchIndex): that is keyed by global cell
//! id and is the wrong axis here.
//!
//! ### A note on PCA
//!
//! Residuals already carry the biological signal as variance, which is the
//! whole point of the transformation. Dividing it back out with
//! `normalise_variance` throws that away, so it should normally be off when
//! PCA runs on a residual source, even though it defaults on elsewhere.

use rayon::prelude::*;
use std::time::Instant;

use indexmap::IndexSet;

use crate::errors::BixverseErrors;
use crate::prelude::*;
use crate::single_cell::sc_data::data_io::{CscGeneChunk, RawCounts, SingleCellReading};

use super::sctransform::stream::SctStreamOpts;

//////////////////////
// ResidualSource   //
//////////////////////

/// A fitted null model that can regenerate any gene's dense residual row.
///
/// Implementors hold one model per group and the per-cell group map, so the
/// consumers below never need to know which method produced the numbers.
pub trait ResidualSource: Sync {
    /// Store gene indices covered, ascending.
    ///
    /// For a grouped source this is the intersection across groups: a gene
    /// that one sample filtered out has no residual there and cannot be on the
    /// shared axis.
    ///
    /// ### Returns
    ///
    /// The gene indices.
    fn genes(&self) -> &[usize];

    /// Position of a store gene index within [`Self::genes`].
    ///
    /// ### Params
    ///
    /// * `gene` - Store gene index.
    ///
    /// ### Returns
    ///
    /// The position, or `None` when the gene is not covered.
    fn position(&self, gene: usize) -> Option<usize> {
        self.genes().binary_search(&gene).ok()
    }

    /// Number of selected cells, which is the length of a residual row.
    ///
    /// ### Returns
    ///
    /// The cell count.
    fn n_cells(&self) -> usize;

    /// Group id per selected cell, in `cell_indices` order.
    ///
    /// ### Returns
    ///
    /// A slice of length [`Self::n_cells`], every entry below
    /// [`Self::n_groups`].
    fn group_of_cell(&self) -> &[u32];

    /// Number of groups, one per fitted model. `1` for a single-sample fit.
    ///
    /// ### Returns
    ///
    /// The group count.
    fn n_groups(&self) -> usize;

    /// Fills `out` with one gene's residuals across every selected cell.
    ///
    /// ### Params
    ///
    /// * `counts` - The gene's non-zero counts.
    /// * `indices` - Cell positions of those counts, within `0..n_cells`.
    /// * `gene_pos` - Position of the gene within [`Self::genes`].
    /// * `out` - Destination row, overwritten in full.
    ///
    /// ### Returns
    ///
    /// `()`, or a [`BixverseErrors`] when `gene_pos` is out of range or the
    /// lengths disagree.
    fn residual_row(
        &self,
        counts: &[f64],
        indices: &[u32],
        gene_pos: usize,
        out: &mut [f32],
    ) -> Result<(), BixverseErrors>;
}

/////////////
// Helpers //
/////////////

/// Validates a group map and reports how many groups it implies.
///
/// Labels must densely cover `0..n_groups`; a gap means a factor was coded
/// without dropping unused levels and some model would be fitted against no
/// cells at all.
///
/// ### Params
///
/// * `group_of_cell` - Group id per selected cell.
/// * `n_cells` - Number of selected cells.
///
/// ### Returns
///
/// The group count, or the matching [`BixverseErrors`] when the map is the
/// wrong length, empty, or leaves a group without cells.
pub fn validate_groups(group_of_cell: &[u32], n_cells: usize) -> Result<usize, BixverseErrors> {
    if group_of_cell.len() != n_cells {
        return Err(BixverseErrors::ResidualGroupLabelLengthMismatch {
            n_labels: group_of_cell.len(),
            n_cells,
        });
    }
    if n_cells == 0 {
        return Err(BixverseErrors::ResidualGroupLabelLengthMismatch {
            n_labels: 0,
            n_cells: 0,
        });
    }

    // A dense label cannot exceed `n_cells - 1`, so capping the tally both
    // bounds the allocation against a nonsense label and, by pigeonhole,
    // guarantees a genuinely empty slot to report when the cap bites.
    let implied = group_of_cell
        .iter()
        .copied()
        .max()
        .map_or(1, |m| m as usize + 1);
    let tally_len = implied.min(n_cells);
    let mut sizes = vec![0_usize; tally_len];
    for &g in group_of_cell {
        let g = g as usize;
        if g < tally_len {
            sizes[g] += 1;
        }
    }

    if let Some(group) = sizes.iter().position(|&s| s == 0) {
        return Err(BixverseErrors::ResidualEmptyGroup {
            group,
            n_groups: implied,
        });
    }

    Ok(implied)
}

/// Intersects per-group gene sets into the shared feature axis.
///
/// Every input must be ascending, which the fitted models guarantee.
///
/// ### Params
///
/// * `per_group` - One ascending gene index set per group.
///
/// ### Returns
///
/// The ascending intersection, or [`BixverseErrors::ResidualEmptyGeneIntersection`]
/// when no gene survives in every group.
pub fn intersect_gene_sets(per_group: &[&[usize]]) -> Result<Vec<usize>, BixverseErrors> {
    let n_groups = per_group.len();
    let Some((first, rest)) = per_group.split_first() else {
        return Err(BixverseErrors::ResidualEmptyGeneIntersection { n_groups });
    };

    let shared: Vec<usize> = first
        .iter()
        .copied()
        .filter(|g| rest.iter().all(|set| set.binary_search(g).is_ok()))
        .collect();

    if shared.is_empty() {
        return Err(BixverseErrors::ResidualEmptyGeneIntersection { n_groups });
    }

    Ok(shared)
}

/// Densifies a gene chunk's raw counts into a `f64` vector of its non-zeros.
///
/// ### Params
///
/// * `chunk` - The gene chunk.
///
/// ### Returns
///
/// The non-zero counts, parallel to `chunk.indices`.
pub fn chunk_counts(chunk: &CscGeneChunk) -> Vec<f64> {
    match &chunk.data_raw {
        RawCounts::U16(v) => v.iter().map(|&x| x as f64).collect(),
        RawCounts::U32(v) => v.iter().map(|&x| x as f64).collect(),
    }
}

/// Per-group sample variance of one residual row, in a single scan.
///
/// The row is written into per-group accumulators rather than split first, so
/// the cost is one pass over the cells whatever the group count.
///
/// ### Params
///
/// * `row` - The residual row, one entry per selected cell.
/// * `group_of_cell` - Group id per selected cell, parallel to `row`.
/// * `n_groups` - Number of groups.
///
/// ### Returns
///
/// One variance per group. A group with fewer than two cells gets `0.0`.
fn grouped_sample_variance(row: &[f32], group_of_cell: &[u32], n_groups: usize) -> Vec<f64> {
    let mut n = vec![0_usize; n_groups];
    let mut sum = vec![0.0_f64; n_groups];
    let mut sum_sq = vec![0.0_f64; n_groups];

    for (&x, &g) in row.iter().zip(group_of_cell) {
        let g = g as usize;
        let x = x as f64;
        n[g] += 1;
        sum[g] += x;
        sum_sq[g] += x * x;
    }

    // The residuals are centred near zero by construction, so the sum of
    // squares never dwarfs the correction term the way it would for raw
    // counts, and the one-pass form is safe here.
    (0..n_groups)
        .map(|g| {
            if n[g] < 2 {
                return 0.0;
            }
            let n_f = n[g] as f64;
            let mean = sum[g] / n_f;
            ((sum_sq[g] - n_f * mean * mean) / (n_f - 1.0)).max(0.0)
        })
        .collect()
}

////////////////////////
// Streaming reducers //
////////////////////////

/// Per-gene residual variance, per group, in one pass over the store.
///
/// This is the HVG ranking statistic: sctransform's
/// `gene_attr$residual_variance` for a scTransform source, and the variance of
/// the analytic Pearson residuals for the other. Each gene's dense row is
/// regenerated, reduced, and dropped, so memory is one row per rayon worker
/// rather than a genes-by-cells matrix.
///
/// ### Params
///
/// * `reader` - Gene-major store.
/// * `source` - The fitted residual source.
/// * `cell_indices` - Global ids of the selected cells, in the order the source
///   was built against.
/// * `opts` - Disk batching and reporting.
///
/// ### Returns
///
/// `n_groups` vectors of one variance per gene in `source.genes()` order, or a
/// [`BixverseErrors`] when the store is cell-major or the selection disagrees
/// with the source.
pub fn residual_variance<S: SingleCellReading>(
    reader: &S,
    source: &dyn ResidualSource,
    cell_indices: &[usize],
    opts: SctStreamOpts,
) -> Result<Vec<Vec<f64>>, BixverseErrors> {
    if !reader.is_gene_based() {
        return Err(BixverseErrors::ReaderModeMismatch {
            actual: "cell-based",
            requested: "gene-based",
        });
    }
    if source.n_cells() != cell_indices.len() {
        return Err(BixverseErrors::LengthMismatch {
            name: "cell_indices",
            expected: source.n_cells(),
            found: cell_indices.len(),
        });
    }

    let verbosity = parse_verbosity_level(opts.verbose);
    let start = Instant::now();

    let genes = source.genes();
    let n_genes = genes.len();
    let n_cells = cell_indices.len();
    let n_groups = source.n_groups();
    let group_of_cell = source.group_of_cell();

    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&c| c as u32).collect();
    let step = opts.gene_batch_size.unwrap_or(n_genes).max(1);

    let mut out = vec![vec![0.0_f64; n_genes]; n_groups];
    let mut done = 0_usize;

    for block in genes.chunks(step) {
        let chunks = reader.read_gene_parallel_filtered(block, &cell_set)?;

        let vars: Vec<(usize, Vec<f64>)> = chunks
            .par_iter()
            .map(|chunk| {
                let pos = source.position(chunk.original_index).ok_or(
                    BixverseErrors::SctGeneNotModelled {
                        gene: chunk.original_index,
                    },
                )?;
                let counts = chunk_counts(chunk);
                let mut row = vec![0.0_f32; n_cells];
                source.residual_row(&counts, &chunk.indices, pos, &mut row)?;
                Ok((pos, grouped_sample_variance(&row, group_of_cell, n_groups)))
            })
            .collect::<Result<Vec<_>, BixverseErrors>>()?;

        for (pos, per_group) in vars {
            for (g, v) in per_group.into_iter().enumerate() {
                out[g][pos] = v;
            }
        }

        let prev = done;
        done += block.len();
        if verbosity.detailed_verbosity() {
            report_decile_progress(done, prev, n_genes, "genes", start.elapsed());
        }
    }

    if verbosity.normal_verbosity() {
        println!(
            "Residual variance for {n_genes} gene(s) across {n_groups} group(s) in {:.2?}",
            start.elapsed()
        );
    }

    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_validate_groups_single_group() {
        let groups = vec![0_u32; 5];
        assert_eq!(validate_groups(&groups, 5).unwrap(), 1);
    }

    #[test]
    fn test_validate_groups_rejects_length_mismatch() {
        let groups = vec![0_u32; 4];
        assert!(matches!(
            validate_groups(&groups, 5),
            Err(BixverseErrors::ResidualGroupLabelLengthMismatch { .. })
        ));
    }

    #[test]
    fn test_validate_groups_rejects_gap() {
        // Group 1 is never used, so a model would be fitted against nothing.
        let groups = vec![0_u32, 0, 2, 2];
        assert!(matches!(
            validate_groups(&groups, 4),
            Err(BixverseErrors::ResidualEmptyGroup { group: 1, .. })
        ));
    }

    #[test]
    fn test_intersect_gene_sets_keeps_shared_only() {
        let a: &[usize] = &[0, 2, 4, 6];
        let b: &[usize] = &[2, 3, 4, 7];
        let c: &[usize] = &[1, 2, 4];
        assert_eq!(intersect_gene_sets(&[a, b, c]).unwrap(), vec![2, 4]);
    }

    #[test]
    fn test_intersect_gene_sets_errors_when_disjoint() {
        let a: &[usize] = &[0, 1];
        let b: &[usize] = &[2, 3];
        assert!(matches!(
            intersect_gene_sets(&[a, b]),
            Err(BixverseErrors::ResidualEmptyGeneIntersection { n_groups: 2 })
        ));
    }

    #[test]
    fn test_grouped_sample_variance_splits_by_group() {
        let row = [1.0_f32, 3.0, 10.0, 20.0];
        let groups = [0_u32, 0, 1, 1];
        let vars = grouped_sample_variance(&row, &groups, 2);
        assert_relative_eq!(vars[0], reference_variance(&row[..2]), epsilon = 1e-9);
        assert_relative_eq!(vars[1], reference_variance(&row[2..]), epsilon = 1e-9);
    }

    /// Two-pass sample variance, the definition the one-pass accumulator has to
    /// agree with.
    fn reference_variance(values: &[f32]) -> f64 {
        let n = values.len() as f64;
        let mean = values.iter().map(|&x| x as f64).sum::<f64>() / n;
        values
            .iter()
            .map(|&x| {
                let d = x as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0)
    }

    #[test]
    fn test_grouped_sample_variance_matches_ungrouped() {
        let row = [1.0_f32, 3.0, 10.0, 20.0];
        let groups = [0_u32; 4];
        let vars = grouped_sample_variance(&row, &groups, 1);
        assert_relative_eq!(vars[0], reference_variance(&row), epsilon = 1e-9);
    }
}
