//! The reader-driven stages.
//!
//! The model needs three sums over the retained count block: per gene, per cell
//! and the grand total. The gene sums and the detection counts that define
//! "retained" come off the gene-major store in one pass. The cell totals then
//! have to be taken over that same retained set, which is a per-cell reduction
//! and so comes off the cell-major store: accumulating them gene-major would
//! need a scratch vector of every cell per worker, while a cell-major sweep is
//! flat in memory and exact.
//!
//! Only then is there anything to "fit", and it is arithmetic, not iteration.
//!
//! ### References
//!
//! Lause, Berens & Kobak, Genome Biology, 2021, 22:258

use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::time::Instant;

use crate::errors::BixverseErrors;
use crate::prelude::*;
use crate::single_cell::sc_data::data_io::{RawCounts, SingleCellReading};
use crate::single_cell::sc_processing::residuals::{distinct_cell_set, validate_groups};
use crate::single_cell::sc_processing::sctransform::stream::SctStreamOpts;

use super::model::{AprModel, AprParams};

//////////////
// GenePass //
//////////////

/// Per-gene summaries over the selected cells, one pass over the gene-major
/// store.
#[derive(Clone, Debug)]
pub struct AprGenePass {
    /// Column sums `sum_c X_cg`, indexed by store gene index.
    pub gene_sums: Vec<f64>,
    /// Number of cells each gene is detected in, indexed by store gene index.
    pub detected: Vec<usize>,
    /// Store indices of the genes passing the `min_cells` filter, ascending.
    pub retained: Vec<usize>,
}

/// Gene sums and detection counts over the selected cells.
///
/// ### Params
///
/// * `reader` - Gene-major store.
/// * `cell_indices` - Global ids of the selected cells.
/// * `params` - Tuning knobs; only `min_cells` is read here.
/// * `opts` - Disk batching and reporting.
///
/// ### Returns
///
/// The pass, or a [`BixverseErrors`] when the store is cell-major, the
/// selection is empty, or no gene passes the filter.
pub fn apr_gene_pass<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    params: &AprParams,
    opts: SctStreamOpts,
) -> Result<AprGenePass, BixverseErrors> {
    if !reader.is_gene_based() {
        return Err(BixverseErrors::ReaderModeMismatch {
            actual: "cell-based",
            requested: "gene-based",
        });
    }
    if cell_indices.is_empty() {
        return Err(BixverseErrors::LengthMismatch {
            name: "cell_indices",
            expected: 1,
            found: 0,
        });
    }

    let verbosity = parse_verbosity_level(opts.verbose);
    let start = Instant::now();

    let n_genes = reader.get_header().total_genes;
    let cell_set = distinct_cell_set(cell_indices)?;
    let step = opts.gene_batch_size.unwrap_or(n_genes).max(1);

    let mut gene_sums = vec![0.0_f64; n_genes];
    let mut detected = vec![0_usize; n_genes];

    for block_start in (0..n_genes).step_by(step) {
        let block_end = (block_start + step).min(n_genes);
        let block: Vec<usize> = (block_start..block_end).collect();
        let chunks = reader.read_gene_parallel_filtered(&block, &cell_set)?;

        let sums: Vec<(usize, f64, usize)> = chunks
            .par_iter()
            .map(|chunk| {
                let sum = match &chunk.data_raw {
                    RawCounts::U16(v) => v.iter().map(|&x| x as f64).sum(),
                    RawCounts::U32(v) => v.iter().map(|&x| x as f64).sum(),
                };
                (chunk.original_index, sum, chunk.indices.len())
            })
            .collect();

        for (g, sum, nnz) in sums {
            gene_sums[g] = sum;
            detected[g] = nnz;
        }

        if verbosity.detailed_verbosity() {
            report_decile_progress(block_end, block_start, n_genes, "genes", start.elapsed());
        }
    }

    let retained: Vec<usize> = (0..n_genes)
        .filter(|&g| detected[g] >= params.min_cells)
        .collect();

    if retained.is_empty() {
        return Err(BixverseErrors::SctNoGenesPassFilter {
            min_cells: params.min_cells,
            n_genes,
        });
    }

    if verbosity.normal_verbosity() {
        println!(
            "Analytic Pearson: swept {n_genes} genes in {:.2?}, {} pass the min_cells filter",
            start.elapsed(),
            retained.len()
        );
    }

    Ok(AprGenePass {
        gene_sums,
        detected,
        retained,
    })
}

/// Per-cell totals restricted to a gene set, off the cell-major store.
///
/// `n_c` in the model is the cell's depth over the genes actually being
/// modelled, not its full library size, so a gene dropped by the `min_cells`
/// filter must not contribute to it.
///
/// ### Params
///
/// * `reader` - Cell-major store.
/// * `cell_indices` - Global ids of the selected cells. The result follows this
///   order.
/// * `genes` - Store indices of the retained genes.
///
/// ### Returns
///
/// One total per selected cell, or a [`BixverseErrors`] when the store is
/// gene-major.
pub fn cell_totals_over_genes<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    genes: &[usize],
) -> Result<Vec<f64>, BixverseErrors> {
    if !reader.is_cell_based() {
        return Err(BixverseErrors::ReaderModeMismatch {
            actual: "gene-based",
            requested: "cell-based",
        });
    }

    // `read_cells_parallel` does not deduplicate, so a repeated cell would be
    // counted twice here while the gene sums, keyed by an `IndexSet`, count it
    // once. `total` would then stop equalling `sum(cell_totals)` and `mu` would
    // no longer be the maximum likelihood solution of anything.
    distinct_cell_set(cell_indices)?;

    let retained: FxHashSet<u32> = genes.iter().map(|&g| g as u32).collect();
    let chunks = reader.read_cells_parallel(cell_indices)?;

    Ok(chunks
        .par_iter()
        .map(|chunk| match &chunk.data_raw {
            RawCounts::U16(v) => chunk
                .indices
                .iter()
                .zip(v)
                .filter(|(g, _)| retained.contains(g))
                .map(|(_, &x)| x as f64)
                .sum(),
            RawCounts::U32(v) => chunk
                .indices
                .iter()
                .zip(v)
                .filter(|(g, _)| retained.contains(g))
                .map(|(_, &x)| x as f64)
                .sum(),
        })
        .collect())
}

///////////
// Model //
///////////

/// Assembles the model from the two passes.
///
/// No iteration: the maximum likelihood solution of the Poisson offset model is
/// the outer product of the marginals over their total (Eq. 3).
///
/// ### Params
///
/// * `pass` - Gene sums and the retained gene set.
/// * `cell_totals` - Per-cell totals over the same retained set.
/// * `params` - Tuning knobs. The clipping range resolves against
///   `cell_totals.len()` when not set.
///
/// ### Returns
///
/// The model, or a [`BixverseErrors`] when theta is non-positive or the block
/// sums to zero.
pub fn build_apr_model(
    pass: &AprGenePass,
    cell_totals: &[f64],
    params: &AprParams,
) -> Result<AprModel, BixverseErrors> {
    params.validate()?;

    let genes = pass.retained.clone();
    let gene_sums: Vec<f64> = genes.iter().map(|&g| pass.gene_sums[g]).collect();
    let total: f64 = gene_sums.iter().sum();

    if total <= 0.0 || total.is_nan() {
        return Err(BixverseErrors::AnalyticPearsonZeroTotal);
    }

    Ok(AprModel {
        genes,
        gene_sums,
        total,
        theta: params.theta,
        clip_range: params.resolve_clip_range(cell_totals.len()),
    })
}

//////////////////
// Grouped fit  //
//////////////////

/// One analytic Pearson model per group, plus the per-cell totals they are
/// scored against.
#[derive(Clone, Debug)]
pub struct AprGroupedFit {
    /// One model per group, in group id order.
    pub models: Vec<AprModel>,
    /// Per-cell totals over the cell's own group's retained genes, in the
    /// selected cell order.
    pub cell_totals: Vec<f64>,
    /// Group id per selected cell, echoed back so the caller can build the
    /// residual source without recomputing it.
    pub group_of_cell: Vec<u32>,
}

/// Fits one analytic Pearson model per group.
///
/// Each group's marginals are its own: a sample with shallower cells gets a
/// smaller `n_c` and its own `p_g`, which is the entire point of doing this per
/// sample rather than once over the pooled matrix.
///
/// The clipping range is the one exception. It is resolved once against the
/// total selected cell count and shared, so residuals from a small sample are
/// not squeezed harder than those from a large one and the two remain
/// comparable when they land side by side in PCA.
///
/// ### Params
///
/// * `gene_reader` - Gene-major store.
/// * `cell_reader` - Cell-major store over the same experiment.
/// * `cell_indices` - Global ids of the selected cells.
/// * `group_of_cell` - Group id per selected cell, densely covering
///   `0..n_groups`.
/// * `params` - Tuning knobs.
/// * `opts` - Disk batching and reporting.
///
/// ### Returns
///
/// The fit, or a [`BixverseErrors`] when the grouping is malformed or a group
/// has no gene passing the filter.
pub fn fit_analytic_pearson_grouped<G: SingleCellReading, C: SingleCellReading>(
    gene_reader: &G,
    cell_reader: &C,
    cell_indices: &[usize],
    group_of_cell: &[u32],
    params: &AprParams,
    opts: SctStreamOpts,
) -> Result<AprGroupedFit, BixverseErrors> {
    params.validate()?;
    let n_groups = validate_groups(group_of_cell, cell_indices.len())?;

    let verbosity = parse_verbosity_level(opts.verbose);
    let start = Instant::now();

    // Shared across groups, so that a small sample is not clipped harder than
    // a large one.
    let shared_params = AprParams {
        clip_range: Some(params.resolve_clip_range(cell_indices.len())),
        ..*params
    };

    let mut models = Vec::with_capacity(n_groups);
    let mut cell_totals = vec![0.0_f64; cell_indices.len()];

    for group in 0..n_groups {
        let slots: Vec<usize> = group_of_cell
            .iter()
            .enumerate()
            .filter(|&(_, &g)| g as usize == group)
            .map(|(slot, _)| slot)
            .collect();
        let group_cells: Vec<usize> = slots.iter().map(|&s| cell_indices[s]).collect();

        let pass = apr_gene_pass(gene_reader, &group_cells, &shared_params, opts)?;
        let totals = cell_totals_over_genes(cell_reader, &group_cells, &pass.retained)?;
        let model = build_apr_model(&pass, &totals, &shared_params)?;

        for (&slot, total) in slots.iter().zip(totals) {
            cell_totals[slot] = total;
        }
        models.push(model);

        if verbosity.normal_verbosity() {
            println!(
                "Analytic Pearson: group {group} fitted over {} cells, {} genes ({:.2?})",
                group_cells.len(),
                models[group].len(),
                start.elapsed()
            );
        }
    }

    Ok(AprGroupedFit {
        models,
        cell_totals,
        group_of_cell: group_of_cell.to_vec(),
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn toy_pass() -> AprGenePass {
        AprGenePass {
            gene_sums: vec![6.0, 9.0, 1.0],
            detected: vec![3, 3, 1],
            retained: vec![0, 1],
        }
    }

    #[test]
    fn test_build_apr_model_uses_retained_genes_only() {
        let pass = toy_pass();
        let model = build_apr_model(&pass, &[5.0, 5.0, 5.0], &AprParams::default()).unwrap();
        assert_eq!(model.genes, vec![0, 1]);
        // Gene 2 is filtered out, so it is not in the total either.
        assert_relative_eq!(model.total, 15.0, epsilon = 1e-12);
        assert_relative_eq!(model.p_gene(0), 0.4, epsilon = 1e-12);
    }

    #[test]
    fn test_build_apr_model_clips_at_sqrt_n() {
        let pass = toy_pass();
        let totals = vec![5.0; 100];
        let model = build_apr_model(&pass, &totals, &AprParams::default()).unwrap();
        assert_relative_eq!(model.clip_range.1, 10.0, epsilon = 1e-12);
    }

    #[test]
    fn test_build_apr_model_rejects_zero_total() {
        let pass = AprGenePass {
            gene_sums: vec![0.0, 0.0],
            detected: vec![5, 5],
            retained: vec![0, 1],
        };
        assert!(matches!(
            build_apr_model(&pass, &[0.0], &AprParams::default()),
            Err(BixverseErrors::AnalyticPearsonZeroTotal)
        ));
    }
}
