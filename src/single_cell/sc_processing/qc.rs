//! Single cell-related QC functions. Checks for example proportion of gene sets
//! or complexity of cells/spots based on the total percentage that the top N
//! genes take.

use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::time::Instant;

use crate::prelude::*;

/// Cells read per batch by the streaming QC passes.
///
/// Larger than the reader default [`CELL_BATCH_SIZE`] because the QC passes
/// only keep a handful of scalars per cell, so a bigger batch amortises the
/// read without growing peak memory much.
const QC_CELL_BATCH_SIZE: usize = 100_000;

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

    let mut results: Vec<Vec<f32>> = Vec::with_capacity(top_n_values.len());

    for &top_n in top_n_values {
        let proportions: Vec<f32> = cell_chunks
            .par_iter()
            .map(|chunk| {
                let mut gene_counts: Vec<u32> = chunk.data_raw.iter().collect();

                if gene_counts.len() <= top_n {
                    1.0
                } else {
                    gene_counts.select_nth_unstable_by(top_n, |a, b| b.cmp(a));
                    let top_sum = gene_counts[..top_n].iter().map(|&x| x as f32).sum::<f32>();
                    top_sum / chunk.library_size as f32
                }
            })
            .collect();

        results.push(proportions);
    }

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

        for (top_idx, &top_n) in top_n_values.iter().enumerate() {
            let proportions: Vec<f32> = cell_chunks
                .par_iter()
                .map(|chunk| {
                    let mut gene_counts: Vec<u32> = chunk.data_raw.iter().collect();

                    if gene_counts.len() <= top_n {
                        1.0
                    } else {
                        gene_counts.select_nth_unstable_by(top_n, |a, b| b.cmp(a));
                        let top_sum = gene_counts[..top_n].iter().map(|&x| x as f32).sum::<f32>();
                        top_sum / chunk.library_size as f32
                    }
                })
                .collect();

            results[top_idx].extend(proportions);
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

    let mut results: Vec<Vec<f32>> = Vec::with_capacity(gene_indices.len());

    for gene_set in gene_indices {
        let hash_gene_set: FxHashSet<&u32> = gene_set.iter().collect();

        let percentage: &Vec<f32> = &cell_chunks
            .par_iter()
            .map(|chunk| {
                let total_sum = chunk
                    .indices
                    .iter()
                    .zip(chunk.data_raw.iter())
                    .filter(|(col_idx, _)| hash_gene_set.contains(col_idx))
                    .map(|(_, val)| val)
                    .sum::<u32>() as f32;
                let lib_size = chunk.library_size as f32;
                total_sum / lib_size
            })
            .collect();

        results.push(percentage.clone());
    }

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
    let hash_gene_sets: Vec<FxHashSet<&u32>> =
        gene_indices.iter().map(|gs| gs.iter().collect()).collect();

    if verbosity.normal_verbosity() {
        println!("Using a streaming approach for gene set percentage calculation.");
    }

    for batch_start in (0..cell_indices.len()).step_by(QC_CELL_BATCH_SIZE) {
        let batch_end = (batch_start + QC_CELL_BATCH_SIZE).min(cell_indices.len());
        let cell_batch = &cell_indices[batch_start..batch_end];

        let cell_chunks = reader.read_cells_parallel(cell_batch)?;

        for (gs_idx, hash_gene_set) in hash_gene_sets.iter().enumerate() {
            let percentage: &Vec<f32> = &cell_chunks
                .par_iter()
                .map(|chunk| {
                    let total_sum = chunk
                        .indices
                        .iter()
                        .zip(chunk.data_raw.iter())
                        .filter(|(col_idx, _)| hash_gene_set.contains(col_idx))
                        .map(|(_, val)| val)
                        .sum::<u32>() as f32;
                    let lib_size = chunk.library_size as f32;
                    total_sum / lib_size
                })
                .collect();
            results[gs_idx].extend(percentage);
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
