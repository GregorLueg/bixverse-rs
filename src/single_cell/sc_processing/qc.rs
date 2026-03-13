//! Single cell-related QC functions. Checks for example proportion of gene sets
//! or complexity of cells/spots based on the total percentage that the top N
//! genes take.

use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::time::Instant;

use crate::prelude::*;

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
/// * `f_path` - File path to the binarised format that contains the cell-based
///   data
/// * `top_n_values` - Slice of top N values to calculate (e.g., &[10, 50, 100])
/// * `cell_indices` - Vector of cell positions to use.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// A vector of vectors with the proportions. Outer vector corresponds to each
/// top_n value, inner vector to each cell.
pub fn get_top_genes_perc(
    f_path: &str,
    top_n_values: &[usize],
    cell_indices: &[usize],
    verbose: bool,
) -> Vec<Vec<f32>> {
    let start_reading = Instant::now();

    let reader = ParallelSparseReader::new(f_path).unwrap();

    let cell_chunks = reader.read_cells_parallel(cell_indices);

    let end_read = start_reading.elapsed();

    if verbose {
        println!("Load in data: {:.2?}", end_read);
    }

    let start_calculations = Instant::now();

    let mut results: Vec<Vec<f32>> = Vec::with_capacity(top_n_values.len());

    for &top_n in top_n_values {
        let proportions: Vec<f32> = cell_chunks
            .par_iter()
            .map(|chunk| {
                let mut gene_counts: Vec<u16> = chunk.data_raw.clone();

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

    if verbose {
        println!(
            "Finished the top genes proportion calculations: {:.2?}",
            end_calculations
        );
    }

    results
}

/// Calculates the cumulative proportion of the top X genes
///
/// Streaming version that reads cells in batches to avoid memory pressure.
///
/// ### Params
///
/// * `f_path` - File path to the binarised format that contains the cell-based
///   data
/// * `top_n_values` - Slice of top N values to calculate (e.g., &[10, 50, 100])
/// * `cell_indices` - Vector of cell positions to use.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// A vector of vectors with the proportions. Outer vector corresponds to each
/// top_n value, inner vector to each cell.
pub fn get_top_genes_perc_streaming(
    f_path: &str,
    top_n_values: &[usize],
    cell_indices: &[usize],
    verbose: bool,
) -> Vec<Vec<f32>> {
    let start_total = Instant::now();

    let reader = ParallelSparseReader::new(f_path).unwrap();

    let mut results: Vec<Vec<f32>> = vec![Vec::new(); top_n_values.len()];

    const CELL_BATCH_SIZE: usize = 100000;

    for batch_start in (0..cell_indices.len()).step_by(CELL_BATCH_SIZE) {
        let batch_end = (batch_start + CELL_BATCH_SIZE).min(cell_indices.len());
        let cell_batch = &cell_indices[batch_start..batch_end];

        let cell_chunks = reader.read_cells_parallel(cell_batch);

        for (top_idx, &top_n) in top_n_values.iter().enumerate() {
            let proportions: Vec<f32> = cell_chunks
                .par_iter()
                .map(|chunk| {
                    let mut gene_counts: Vec<u16> = chunk.data_raw.clone();

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

        if verbose && batch_start % (CELL_BATCH_SIZE * 5) == 0 {
            let progress = ((batch_start + 1) as f32 / cell_indices.len() as f32) * 100.0;
            println!(
                " Reading cells and calculating proportions: {:.1}%",
                progress
            );
        }
    }

    let end_total = start_total.elapsed();

    if verbose {
        println!(
            "Finished the top genes proportion calculations: {:.2?}",
            end_total
        );
    }

    results
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
/// * `f_path` - File path to the binarised format that contains the cell-based
///   data
/// * `gene_indices` - Vector of index positions of the genes of interest
/// * `cell_indices` - Vector of cell positions to use.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// A vector with the percentages of these genes over the total reads.
pub fn get_gene_set_perc(
    f_path: &str,
    gene_indices: Vec<Vec<u16>>,
    cell_indices: &[usize],
    verbose: bool,
) -> Vec<Vec<f32>> {
    let start_reading = Instant::now();

    let reader = ParallelSparseReader::new(f_path).unwrap();

    let cell_chunks = reader.read_cells_parallel(cell_indices);

    let end_read = start_reading.elapsed();

    if verbose {
        println!("Load in data: {:.2?}", end_read);
    }

    let start_calculations = Instant::now();

    let mut results: Vec<Vec<f32>> = Vec::with_capacity(gene_indices.len());

    for gene_set in gene_indices {
        let hash_gene_set: FxHashSet<&u16> = gene_set.iter().collect();

        let percentage: &Vec<f32> = &cell_chunks
            .par_iter()
            .map(|chunk| {
                let total_sum = chunk
                    .indices
                    .iter()
                    .zip(&chunk.data_raw)
                    .filter(|(col_idx, _)| hash_gene_set.contains(col_idx))
                    .map(|(_, val)| val)
                    .sum::<u16>() as f32;
                let lib_size = chunk.library_size as f32;
                total_sum / lib_size
            })
            .collect();

        results.push(percentage.clone());
    }

    let end_calculations = start_calculations.elapsed();

    if verbose {
        println!(
            "Finished the gene set proportion calculations: {:.2?}",
            end_calculations
        );
    }

    results
}

/// Calculates the percentage within the gene set(s)
///
/// Helper function to calculate QC metrics such as mitochondrial proportions,
/// ribosomal proportions, etc. This function implements streaming and reads in
/// the cells in chunks to avoid memory pressure.
///
/// ### Params
///
/// * `f_path` - File path to the binarised format that contains the cell-based
///   data
/// * `gene_indices` - Vector of index positions of the genes of interest
/// * `cell_indices` - Vector of cell positions to use.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// A vector with the percentages of these genes over the total reads.
pub fn get_gene_set_perc_streaming(
    f_path: &str,
    gene_indices: Vec<Vec<u16>>,
    cell_indices: &[usize],
    verbose: bool,
) -> Vec<Vec<f32>> {
    let start_total = Instant::now();

    let reader = ParallelSparseReader::new(f_path).unwrap();

    let mut results: Vec<Vec<f32>> = vec![Vec::new(); gene_indices.len()];
    let hash_gene_sets: Vec<FxHashSet<&u16>> =
        gene_indices.iter().map(|gs| gs.iter().collect()).collect();

    const CELL_BATCH_SIZE: usize = 100000;

    for batch_start in (0..cell_indices.len()).step_by(CELL_BATCH_SIZE) {
        let batch_end = (batch_start + CELL_BATCH_SIZE).min(cell_indices.len());
        let cell_batch = &cell_indices[batch_start..batch_end];

        let cell_chunks = reader.read_cells_parallel(cell_batch);

        for (gs_idx, hash_gene_set) in hash_gene_sets.iter().enumerate() {
            let percentage: &Vec<f32> = &cell_chunks
                .par_iter()
                .map(|chunk| {
                    let total_sum = chunk
                        .indices
                        .iter()
                        .zip(&chunk.data_raw)
                        .filter(|(col_idx, _)| hash_gene_set.contains(col_idx))
                        .map(|(_, val)| val)
                        .sum::<u16>() as f32;
                    let lib_size = chunk.library_size as f32;
                    total_sum / lib_size
                })
                .collect();
            results[gs_idx].extend(percentage);
        }

        if verbose && batch_start % (CELL_BATCH_SIZE * 5) == 0 {
            let progress = ((batch_start + 1) as f32 / cell_indices.len() as f32) * 100.0;
            println!(
                " Reading cells and calculating proportions: {:.1}%",
                progress
            );
        }
    }

    let end_total = start_total.elapsed();

    if verbose {
        println!(
            "Finished the gene set proportion calculations: {:.2?}",
            end_total
        );
    }

    results
}
