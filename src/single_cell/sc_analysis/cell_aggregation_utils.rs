use faer::Mat;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::time::Instant;

use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Helper function to aggregate the meta cells
///
/// The function will generate the metacells based on the provided indices.
/// Per meta-cell it will aggregate the raw counts and recalculate the norm
/// counts based on the aggregated counts.
///
/// ### Params
///
/// * `reader` - The reader structure to get the cells from disk
/// * `meta_cells` - The indices of the meta cells
/// * `target_size` - Float defining the target size for the normalisation
///   procedure. Usually defaults to `1e4` in single cell.
/// * `n_genes` - Total number of genes in the data
///
/// ### Return
///
/// `CompressedSparseData` in CSR format with aggregated raw counts and re-
/// normalised counts per meta cell.
pub fn aggregate_meta_cells(
    reader: &ParallelSparseReader,
    metacells: &[&[usize]],
    target_size: f32,
    n_genes: usize,
) -> CompressedSparseData<u32, f32> {
    let n_metacells = metacells.len();
    let mut all_data: Vec<u32> = Vec::new();
    let mut all_data_norm: Vec<f32> = Vec::new();
    let mut all_indices: Vec<usize> = Vec::new();
    let mut all_indptr: Vec<usize> = vec![0];

    const CHUNK_SIZE: usize = 1000;

    for chunk_start in (0..n_metacells).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(n_metacells);
        let chunk = &metacells[chunk_start..chunk_end];

        let results: Vec<(Vec<usize>, Vec<u32>, Vec<f32>)> = chunk
            .par_iter()
            .map(|cell_idx| {
                let cells = reader.read_cells_parallel(cell_idx);
                let mut gene_counts: FxHashMap<usize, u32> = FxHashMap::default();
                let mut library_size: u32 = 0;

                for cell in &cells {
                    for (idx, &count) in cell.indices.iter().zip(cell.data_raw.iter()) {
                        *gene_counts.entry(*idx as usize).or_insert(0) += count as u32;
                        library_size += count as u32;
                    }
                }

                // sort for CSR format
                let mut entries: Vec<(usize, u32)> = gene_counts.into_iter().collect();
                entries.sort_by_key(|(idx, _)| *idx);

                let indices: Vec<usize> = entries.iter().map(|(idx, _)| *idx).collect();
                let raw_counts: Vec<u32> = entries.iter().map(|(_, count)| *count).collect();
                let norm_counts: Vec<f32> = entries
                    .iter()
                    .map(|(_, count)| {
                        let norm = (*count as f32 / library_size as f32) * target_size;
                        (norm + 1.0).ln()
                    })
                    .collect();

                (indices, raw_counts, norm_counts)
            })
            .collect();

        for (indices, raw_counts, norm_counts) in results {
            all_indices.extend(indices);
            all_data.extend(raw_counts);
            all_data_norm.extend(norm_counts);
            all_indptr.push(all_indices.len());
        }
    }

    CompressedSparseData::new_csr(
        &all_data,
        &all_indices,
        &all_indptr,
        Some(&all_data_norm),
        (n_metacells, n_genes),
    )
}

/// Convert metacell groups to flat assignments, handling unassigned cells
///
/// ### Params
///
/// * `metacells` - Vector of cell groups (metacell → [cells])
/// * `n_cells` - Total number of cells
///
/// ### Returns
///
/// Flat assignment vector where assignments[cell_id] = Some(metacell_id)
/// or None if cell is unassigned
pub fn metacells_to_assignments(metacells: &[&[usize]], n_cells: usize) -> Vec<Option<usize>> {
    let mut assignments = vec![None; n_cells];

    for (metacell_id, &cells) in metacells.iter().enumerate() {
        for &cell_id in cells {
            if cell_id < n_cells {
                assignments[cell_id] = Some(metacell_id);
            }
        }
    }

    assignments
}

////////////////////
// Pseudo-bulking //
////////////////////

/// Enum for Pseudo-bulking
#[derive(Debug, Clone, Default)]
pub enum PseudoBulk {
    #[default]
    /// Shall raw counts be pseudo-bulked
    Raw,
    /// Shall normalised counts be pseudo-bulked
    Norm,
}

/// Helper function to parse pseudo-bulk type
///
/// ### Params
///
/// * `s` - Type of pseudo-bulk to perform
///
/// ### Returns
///
/// Option of the PseudoBulk enum
pub fn parse_pseudo_bulk(s: &str) -> Option<PseudoBulk> {
    match s.to_lowercase().as_str() {
        "raw" => Some(PseudoBulk::Raw),
        "norm" | "normalised" | "normalized" => Some(PseudoBulk::Norm),
        _ => None,
    }
}

/// Pseudo-bulk data across cells based on cell indices (dense output)
///
/// ### Params
///
/// * `f_path` - File path to the cell-based binary file.
/// * `cell_indices` - Slice of indices to pseudo-bulk.
/// * `bulk_type` - Whether to pseudo-bulk raw (sum) or normalised (average)
///   counts.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// Dense matrix of samples x genes pseudo-bulked.
pub fn get_pseudo_bulked_counts_dense(
    f_path: &str,
    cell_indices: &[Vec<usize>],
    bulk_type: PseudoBulk,
    verbose: bool,
) -> Mat<f64> {
    let reader = ParallelSparseReader::new(f_path).unwrap();
    let n_genes = reader.get_header().total_genes;
    let n_groups = cell_indices.len();
    let mut result = Mat::zeros(n_groups, n_genes);

    for (group_idx, indices) in cell_indices.iter().enumerate() {
        let start_group = Instant::now();
        let chunks = reader.read_cells_parallel(indices);
        let n_cells = indices.len() as f64;

        for chunk in chunks {
            match bulk_type {
                PseudoBulk::Raw => {
                    for (value, &gene_idx) in chunk.data_raw.iter().zip(chunk.indices.iter()) {
                        result[(group_idx, gene_idx as usize)] += *value as f64;
                    }
                }
                PseudoBulk::Norm => {
                    for (value, &gene_idx) in chunk.data_norm.iter().zip(chunk.indices.iter()) {
                        result[(group_idx, gene_idx as usize)] += value.to_f64();
                    }
                }
            }
        }

        if matches!(bulk_type, PseudoBulk::Norm) {
            for gene_idx in 0..n_genes {
                result[(group_idx, gene_idx)] /= n_cells;
            }
        }

        if verbose && (group_idx + 1) % 10 == 0 {
            let elapsed = start_group.elapsed();
            let pct_complete = ((group_idx + 1) as f32 / n_groups as f32) * 100.0;
            println!(
                "Processed group {} out of {} (took {:.2?}, completed {:.1}%)",
                group_idx + 1,
                n_groups,
                elapsed,
                pct_complete
            );
        }
    }

    result
}

/// Pseudo-bulk data across cells based on cell indices (sparse CSR output)
///
/// ### Params
///
/// * `f_path` - File path to the cell-based binary file.
/// * `cell_indices` - Slice of indices to pseudo-bulk.
/// * `bulk_type` - Whether to pseudo-bulk raw (sum) or normalised (average)
///   counts.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// Sparse CSR matrix of samples x genes pseudo-bulked.
pub fn get_pseudo_bulked_counts_sparse(
    f_path: &str,
    cell_indices: &[Vec<usize>],
    bulk_type: PseudoBulk,
    verbose: bool,
) -> CompressedSparseData<f64> {
    let reader = ParallelSparseReader::new(f_path).unwrap();
    let n_genes = reader.get_header().total_genes;
    let n_groups = cell_indices.len();
    let mut row_data: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); n_groups];

    for (group_idx, indices) in cell_indices.iter().enumerate() {
        let start_group = Instant::now();
        let chunks = reader.read_cells_parallel(indices);
        let n_cells = indices.len() as f64;

        for chunk in chunks {
            match bulk_type {
                PseudoBulk::Raw => {
                    for (value, &gene_idx) in chunk.data_raw.iter().zip(chunk.indices.iter()) {
                        *row_data[group_idx].entry(gene_idx as usize).or_insert(0.0) +=
                            *value as f64;
                    }
                }
                PseudoBulk::Norm => {
                    for (value, &gene_idx) in chunk.data_norm.iter().zip(chunk.indices.iter()) {
                        *row_data[group_idx].entry(gene_idx as usize).or_insert(0.0) +=
                            value.to_f64();
                    }
                }
            }
        }

        if matches!(bulk_type, PseudoBulk::Norm) {
            for value in row_data[group_idx].values_mut() {
                *value /= n_cells;
            }
        }

        if verbose && (group_idx + 1) % 10 == 0 {
            let elapsed = start_group.elapsed();
            let pct_complete = ((group_idx + 1) as f32 / n_groups as f32) * 100.0;
            println!(
                "Processed group {} out of {} (took {:.2?}, completed {:.1}%)",
                group_idx + 1,
                n_groups,
                elapsed,
                pct_complete
            );
        }
    }

    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0];

    for row_map in row_data {
        let mut sorted_entries: Vec<_> = row_map.into_iter().collect();
        sorted_entries.sort_by_key(|(idx, _)| *idx);

        for (idx, value) in sorted_entries {
            data.push(value);
            indices.push(idx);
        }
        indptr.push(data.len());
    }

    CompressedSparseData {
        data,
        indices,
        indptr,
        cs_type: CompressedSparseFormat::Csr,
        data_2: None,
        shape: (n_groups, n_genes),
    }
}
