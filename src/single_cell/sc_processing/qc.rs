//! Single cell-related QC functions. Checks for example proportion of gene sets
//! or complexity of cells/spots based on the total percentage that the top N
//! genes take.

use rayon::prelude::*;
use std::time::Instant;

use crate::prelude::*;
use crate::single_cell::sc_data::data_io::CsrCellChunk;

/// Cells read per batch by the streaming QC passes.
///
/// Larger than the reader default [`CELL_BATCH_SIZE`] because the QC passes
/// only keep a handful of scalars per cell, so a bigger batch amortises the
/// read without growing peak memory much.
const QC_CELL_BATCH_SIZE: usize = 100_000;

/// Fraction of a barcode's library taken by some subset of its counts.
///
/// Guards the zero-library case. A barcode with no counts at all is reachable
/// whenever the store was ingested with `min_lib_size = 0`, which is what the
/// CellSweep workflow requires so the empty droplets survive. Returning `0.0`
/// rather than `NaN` matters because these metrics feed MAD outlier detection:
/// a single `NaN` propagates through the median and the MAD and silently
/// invalidates the calls for every cell, whereas `0.0` flags the offending
/// barcode as the low outlier it is.
///
/// ### Params
///
/// * `numerator` - Counts in the subset of interest.
/// * `library_size` - Total counts in the barcode.
///
/// ### Returns
///
/// The fraction, or `0.0` for an empty barcode.
#[inline]
fn fraction_of_library(numerator: f32, library_size: usize) -> f32 {
    if library_size == 0 {
        0.0
    } else {
        numerator / library_size as f32
    }
}

/// Proportion of each cell's library taken by its top N genes, for several N
///
/// Each cell is collected once and the N values are handled in descending
/// order, each `select_nth` running on the prefix the previous one left.
///
/// ### Params
///
/// * `cell_chunks` - The cells.
/// * `top_n_values` - The N values, in the order the output rows must take.
///
/// ### Returns
///
/// One vector of per-cell proportions per N value.
fn top_genes_fractions(cell_chunks: &[CsrCellChunk], top_n_values: &[usize]) -> Vec<Vec<f32>> {
    let n_top = top_n_values.len();
    if n_top == 0 {
        return Vec::new();
    }

    let mut descending: Vec<usize> = (0..n_top).collect();
    descending.sort_unstable_by(|&a, &b| top_n_values[b].cmp(&top_n_values[a]));

    let mut flat = vec![0.0_f32; cell_chunks.len() * n_top];

    flat.par_chunks_mut(n_top)
        .zip(cell_chunks.par_iter())
        .for_each_init(Vec::<u32>::new, |gene_counts, (out, chunk)| {
            gene_counts.clear();
            gene_counts.extend(chunk.data_raw.iter());
            let mut bound = gene_counts.len();

            for &slot in &descending {
                let top_n = top_n_values[slot];
                out[slot] = if gene_counts.len() <= top_n {
                    1.0
                } else {
                    // equal N values reuse the partition already made
                    if top_n < bound {
                        gene_counts[..bound].select_nth_unstable_by(top_n, |a, b| b.cmp(a));
                        bound = top_n;
                    }
                    let top_sum = gene_counts[..top_n].iter().map(|&x| x as f32).sum::<f32>();
                    fraction_of_library(top_sum, chunk.library_size)
                };
            }
        });

    (0..n_top)
        .map(|slot| flat.iter().skip(slot).step_by(n_top).copied().collect())
        .collect()
}

/// Dense membership mask of a gene set
///
/// ### Params
///
/// * `gene_set` - Gene indices in the set.
/// * `n_genes` - Genes in the store; ids at or above it cannot occur in a
///   cell and are ignored.
///
/// ### Returns
///
/// The mask, indexed by gene.
fn gene_set_mask(gene_set: &[u32], n_genes: usize) -> Vec<bool> {
    let mut mask = vec![false; n_genes];
    for &g in gene_set {
        if let Some(slot) = mask.get_mut(g as usize) {
            *slot = true;
        }
    }
    mask
}

/// Proportion of each cell's library taken by one gene set
///
/// ### Params
///
/// * `cell_chunks` - The cells.
/// * `mask` - Membership mask from [gene_set_mask].
///
/// ### Returns
///
/// Per-cell proportions.
fn gene_set_fractions(cell_chunks: &[CsrCellChunk], mask: &[bool]) -> Vec<f32> {
    cell_chunks
        .par_iter()
        .map(|chunk| {
            let total_sum = chunk
                .indices
                .iter()
                .zip(chunk.data_raw.iter())
                .filter(|(col_idx, _)| mask.get(**col_idx as usize).copied().unwrap_or(false))
                .map(|(_, val)| val)
                .sum::<u32>() as f32;
            fraction_of_library(total_sum, chunk.library_size)
        })
        .collect()
}

///////////////////////////////////////////
// QC metrics based on cumulative counts //
///////////////////////////////////////////

/// Calculates the cumulative proportion of the top X genes
///
/// Helper function to assess cell quality/complexity by measuring how much
/// of the total counts are concentrated in the most highly expressed genes.
///
/// ### Params
///
/// * `reader` - Reader over the cell-based count store.
/// * `top_n_values` - Slice of top N values to calculate (e.g., &[10, 50, 100])
/// * `cell_indices` - Vector of cell positions to use.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A vector of vectors with the proportions. Outer vector corresponds to each
/// top_n value, inner vector to each cell.
pub fn get_top_genes_perc<S: SingleCellReading>(
    reader: &S,
    top_n_values: &[usize],
    cell_indices: &[usize],
    verbose: usize,
) -> Result<Vec<Vec<f32>>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    let start_reading = Instant::now();

    let cell_chunks = reader.read_cells_parallel(cell_indices)?;

    let end_read = start_reading.elapsed();

    if verbosity.normal_verbosity() {
        println!("Load in data: {:.2?}", end_read);
    }

    let start_calculations = Instant::now();

    let results = top_genes_fractions(&cell_chunks, top_n_values);

    let end_calculations = start_calculations.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "Finished the top genes proportion calculations: {:.2?}",
            end_calculations
        );
    }

    Ok(results)
}

/// Calculates the cumulative proportion of the top X genes
///
/// Streaming version that reads cells in batches to avoid memory pressure.
///
/// ### Params
///
/// * `reader` - Reader over the cell-based count store.
/// * `top_n_values` - Slice of top N values to calculate (e.g., &[10, 50, 100])
/// * `cell_indices` - Vector of cell positions to use.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A vector of vectors with the proportions. Outer vector corresponds to each
/// top_n value, inner vector to each cell.
pub fn get_top_genes_perc_streaming<S: SingleCellReading>(
    reader: &S,
    top_n_values: &[usize],
    cell_indices: &[usize],
    verbose: usize,
) -> Result<Vec<Vec<f32>>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    let start_total = Instant::now();

    let mut results: Vec<Vec<f32>> = vec![Vec::new(); top_n_values.len()];

    if verbosity.normal_verbosity() {
        println!("Using a streaming approach for top gene percentage calculations.");
    }

    for batch_start in (0..cell_indices.len()).step_by(QC_CELL_BATCH_SIZE) {
        let batch_end = (batch_start + QC_CELL_BATCH_SIZE).min(cell_indices.len());
        let cell_batch = &cell_indices[batch_start..batch_end];

        let cell_chunks = reader.read_cells_parallel(cell_batch)?;

        for (slot, proportions) in top_genes_fractions(&cell_chunks, top_n_values)
            .into_iter()
            .enumerate()
        {
            results[slot].extend(proportions);
        }

        if verbosity.detailed_verbosity() {
            report_decile_progress(
                batch_end,
                batch_start,
                cell_indices.len(),
                "cells",
                start_total.elapsed(),
            );
        }
    }

    let end_total = start_total.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "Finished the top genes proportion calculations: {:.2?}",
            end_total
        );
    }

    Ok(results)
}

///////////////////////////////
// QC metrics based on genes //
///////////////////////////////

/// Calculates the percentage within the gene set(s)
///
/// Helper function to calculate QC metrics such as mitochondrial proportions,
/// ribosomal proportions, etc.
///
/// ### Params
///
/// * `reader` - Reader over the cell-based count store.
/// * `gene_indices` - Vector of index positions of the genes of interest
/// * `cell_indices` - Vector of cell positions to use.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A vector with the percentages of these genes over the total reads.
pub fn get_gene_set_perc<S: SingleCellReading>(
    reader: &S,
    gene_indices: Vec<Vec<u32>>,
    cell_indices: &[usize],
    verbose: usize,
) -> Result<Vec<Vec<f32>>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    let start_reading = Instant::now();

    let cell_chunks = reader.read_cells_parallel(cell_indices)?;

    let end_read = start_reading.elapsed();

    if verbosity.normal_verbosity() {
        println!("Load in data: {:.2?}", end_read);
    }

    let start_calculations = Instant::now();

    let n_genes = reader.get_header().total_genes;
    let results: Vec<Vec<f32>> = gene_indices
        .iter()
        .map(|gene_set| gene_set_fractions(&cell_chunks, &gene_set_mask(gene_set, n_genes)))
        .collect();

    let end_calculations = start_calculations.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "Finished the gene set proportion calculations: {:.2?}",
            end_calculations
        );
    }

    Ok(results)
}

/// Calculates the percentage within the gene set(s)
///
/// Helper function to calculate QC metrics such as mitochondrial proportions,
/// ribosomal proportions, etc. This function implements streaming and reads in
/// the cells in chunks to avoid memory pressure.
///
/// ### Params
///
/// * `reader` - Reader over the cell-based count store.
/// * `gene_indices` - Vector of index positions of the genes of interest
/// * `cell_indices` - Vector of cell positions to use.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A vector with the percentages of these genes over the total reads.
pub fn get_gene_set_perc_streaming<S: SingleCellReading>(
    reader: &S,
    gene_indices: Vec<Vec<u32>>,
    cell_indices: &[usize],
    verbose: usize,
) -> Result<Vec<Vec<f32>>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    let start_total = Instant::now();

    let mut results: Vec<Vec<f32>> = vec![Vec::new(); gene_indices.len()];
    let n_genes = reader.get_header().total_genes;
    let masks: Vec<Vec<bool>> = gene_indices
        .iter()
        .map(|gs| gene_set_mask(gs, n_genes))
        .collect();

    if verbosity.normal_verbosity() {
        println!("Using a streaming approach for gene set percentage calculation.");
    }

    for batch_start in (0..cell_indices.len()).step_by(QC_CELL_BATCH_SIZE) {
        let batch_end = (batch_start + QC_CELL_BATCH_SIZE).min(cell_indices.len());
        let cell_batch = &cell_indices[batch_start..batch_end];

        let cell_chunks = reader.read_cells_parallel(cell_batch)?;

        for (gs_idx, mask) in masks.iter().enumerate() {
            results[gs_idx].extend(gene_set_fractions(&cell_chunks, mask));
        }

        if verbosity.detailed_verbosity() {
            report_decile_progress(
                batch_end,
                batch_start,
                cell_indices.len(),
                "cells",
                start_total.elapsed(),
            );
        }
    }

    let end_total = start_total.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "Finished the gene set proportion calculations: {:.2?}",
            end_total
        );
    }

    Ok(results)
}
