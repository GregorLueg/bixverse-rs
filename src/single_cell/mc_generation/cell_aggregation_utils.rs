//! Helpers around pseudo-bulking and meta-cell aggregations of single cells.

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
/// `CompressedSparseData2` in CSR format with aggregated raw counts and re-
/// normalised counts per meta cell.
pub fn aggregate_meta_cells<S: SingleCellReading>(
    reader: &S,
    metacells: &[&[usize]],
    target_size: f32,
    n_genes: usize,
) -> Result<CompressedSparseData2<u32, f32>, BixverseErrors> {
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
            .map(|cell_idx| -> Result<_, BixverseErrors> {
                let cells = reader.read_cells_parallel(cell_idx)?;
                let mut gene_counts: FxHashMap<usize, u32> = FxHashMap::default();
                let mut library_size: u32 = 0;
                for cell in &cells {
                    for (idx, count) in cell.indices.iter().zip(cell.data_raw.iter()) {
                        *gene_counts.entry(*idx as usize).or_insert(0) += count;
                        library_size += count;
                    }
                }
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
                Ok((indices, raw_counts, norm_counts))
            })
            .collect::<Result<Vec<_>, _>>()?;

        for (indices, raw_counts, norm_counts) in results {
            all_indices.extend(indices);
            all_data.extend(raw_counts);
            all_data_norm.extend(norm_counts);
            all_indptr.push(all_indices.len());
        }
    }

    Ok(CompressedSparseData2::new_csr(
        &all_data,
        &all_indices.index_cast(),
        &all_indptr.index_cast(),
        Some(&all_data_norm),
        (n_metacells, n_genes),
    ))
}

/// Convert metacell groups to flat assignments, handling unassigned cells
///
/// ### Params
///
/// * `metacells` - Vector of cell groups (metacell → `[cells]`)
/// * `n_cells` - Total number of cells
///
/// ### Returns
///
/// Flat assignment vector where `assignments[cell_id] = Some(metacell_id)`
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

/// Remap subset assignments back to original index space
///
/// Takes metacell assignments computed on a subset of cells and maps them
/// back to the full original index space. Cells not in the subset will have
/// `None` assignments.
///
/// ### Params
///
/// * `subset_assignments` - Vector of metacell assignments in subset index space
/// * `subset_to_orig` - Mapping from subset indices to original indices
/// * `n_total` - Total number of cells in original space
///
/// ### Return
///
/// Vector of assignments in original index space with `None` for cells not
/// in the subset
pub fn remap_assignments_to_original(
    subset_assignments: &[Option<usize>],
    subset_to_orig: &[usize],
    n_total: usize,
) -> Vec<Option<usize>> {
    let mut full_assignments = vec![None; n_total];
    for (subset_idx, &metacell_id) in subset_assignments.iter().enumerate() {
        if let Some(orig_idx) = subset_to_orig.get(subset_idx) {
            full_assignments[*orig_idx] = metacell_id;
        }
    }
    full_assignments
}

/// Remap metacell indices from subset space to original space
///
/// Takes metacell groups where cell indices are in subset space and transforms
/// all indices back to original space. This is used when metacells are computed
/// on a subset of cells but need to be aggregated from the full dataset.
///
/// ### Params
///
/// * `metacells` - Vector of metacell groups with cell indices in subset space
/// * `subset_to_orig` - Mapping from subset indices to original indices
///
/// ### Return
///
/// Vector of metacell groups with cell indices in original space
pub fn remap_metacells_to_original(
    metacells: &[&[usize]],
    subset_to_orig: &[usize],
) -> Vec<Vec<usize>> {
    metacells
        .iter()
        .map(|&cells| cells.iter().map(|&idx| subset_to_orig[idx]).collect())
        .collect()
}

////////////////////
// Pseudo-bulking //
////////////////////

/// Gene-major batch size for [pseudo_bulk_genes_dense].
///
/// One batch holds `GENE_PSEUDO_BULK_BATCH * n_groups` accumulators plus the
/// chunks themselves, so this bounds the transient memory rather than the
/// result, which is dense and sized by the caller's gene list.
const GENE_PSEUDO_BULK_BATCH: usize = 1000;

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
/// * `reader` - Reader for the cell-based store.
/// * `cell_indices` - Slice of indices to pseudo-bulk.
/// * `bulk_type` - Whether to pseudo-bulk raw (sum) or normalised (average)
///   counts.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Dense matrix of samples x genes pseudo-bulked.
pub fn get_pseudo_bulked_counts_dense<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[Vec<usize>],
    bulk_type: PseudoBulk,
    verbose: usize,
) -> Result<Mat<f64>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    let n_genes = reader.get_header().total_genes;
    let n_groups = cell_indices.len();
    let mut result = Mat::zeros(n_groups, n_genes);

    for (group_idx, indices) in cell_indices.iter().enumerate() {
        let start_group = Instant::now();
        let chunks = reader.read_cells_parallel(indices)?;
        let n_cells = indices.len() as f64;

        for chunk in chunks {
            match bulk_type {
                PseudoBulk::Raw => {
                    for (value, &gene_idx) in chunk.data_raw.iter().zip(chunk.indices.iter()) {
                        result[(group_idx, gene_idx as usize)] += value as f64;
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

        if verbosity.normal_verbosity() && (group_idx + 1) % 10 == 0 {
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

    Ok(result)
}

/// Pseudo-bulk data across cells based on cell indices (sparse CSR output)
///
/// ### Params
///
/// * `reader` - Reader for the cell-based store.
/// * `cell_indices` - Slice of indices to pseudo-bulk.
/// * `bulk_type` - Whether to pseudo-bulk raw (sum) or normalised (average)
///   counts.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Sparse CSR matrix of samples x genes pseudo-bulked.
pub fn get_pseudo_bulked_counts_sparse<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[Vec<usize>],
    bulk_type: PseudoBulk,
    verbose: usize,
) -> Result<CompressedSparseData2<f64>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    let n_genes = reader.get_header().total_genes;
    let n_groups = cell_indices.len();
    let mut row_data: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); n_groups];

    for (group_idx, indices) in cell_indices.iter().enumerate() {
        let start_group = Instant::now();
        let chunks = reader.read_cells_parallel(indices)?;
        let n_cells = indices.len() as f64;

        for chunk in chunks {
            match bulk_type {
                PseudoBulk::Raw => {
                    for (value, &gene_idx) in chunk.data_raw.iter().zip(chunk.indices.iter()) {
                        *row_data[group_idx].entry(gene_idx as usize).or_insert(0.0) +=
                            value as f64;
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

        if verbosity.normal_verbosity() && (group_idx + 1) % 10 == 0 {
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

    Ok(CompressedSparseData2 {
        data,
        indices: indices.index_cast(),
        indptr: indptr.index_cast(),
        cs_type: CompressedSparseFormat::Csr,
        data_2: None,
        shape: (n_groups, n_genes),
    })
}

/// Gene-major pseudo-bulk over a subset of genes.
///
/// The twin of [get_pseudo_bulked_counts_dense], which reads cell chunks and
/// therefore needs a cell-major store and always sweeps every gene. This one
/// reads gene chunks, so it runs on an [crate::single_cell::sc_data::in_memory_io::InMemorySparseReader]
/// as happily as on a file, and it only touches the genes asked for. Both
/// matter to anything that has already narrowed down to a signature.
///
/// Groups may overlap; a cell contributing to several is counted in each. The
/// inverse map from cells to groups is built once as a flat CSR rather than a
/// map per cell.
///
/// ### Params
///
/// * `reader` - Gene-major reader
/// * `gene_indices` - Genes to aggregate. Result rows follow this order.
/// * `groups` - Cell indices per group. Result columns follow this order.
/// * `bulk_type` - [PseudoBulk::Raw] sums raw counts, [PseudoBulk::Norm]
///   averages the normalised layer over every cell in the group, zeros
///   included, which is what an R `colMeans` over a dense block gives.
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// An `n_genes x n_groups` matrix, or [BixverseErrors::InvalidArgument] for an
/// out-of-range cell index or an empty group.
pub fn pseudo_bulk_genes_dense<S: SingleCellReading>(
    reader: &S,
    gene_indices: &[usize],
    groups: &[Vec<usize>],
    bulk_type: PseudoBulk,
    verbose: usize,
) -> Result<Mat<f64>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let n_cells = reader.get_header().total_cells;
    let n_groups = groups.len();
    let n_genes = gene_indices.len();

    if n_groups == 0 || n_genes == 0 {
        return Ok(Mat::zeros(n_genes, n_groups));
    }

    // Flat CSR of cell -> groups. Counting pass, prefix sum, fill.
    let mut per_cell = vec![0_u32; n_cells + 1];
    for (g, members) in groups.iter().enumerate() {
        if members.is_empty() {
            return Err(BixverseErrors::InvalidArgument(format!(
                "pseudo-bulk group {g} has no cells."
            )));
        }
        for &c in members.iter() {
            if c >= n_cells {
                return Err(BixverseErrors::InvalidArgument(format!(
                    "cell index {c} is outside 0..{n_cells}."
                )));
            }
            per_cell[c + 1] += 1;
        }
    }
    for i in 0..n_cells {
        per_cell[i + 1] += per_cell[i];
    }
    let offsets = per_cell;
    let mut cursor = offsets.clone();
    let mut membership = vec![0_u32; offsets[n_cells] as usize];
    for (g, members) in groups.iter().enumerate() {
        for &c in members.iter() {
            membership[cursor[c] as usize] = g as u32;
            cursor[c] += 1;
        }
    }

    let sizes: Vec<f64> = groups.iter().map(|m| m.len() as f64).collect();
    let mut out = Mat::<f64>::zeros(n_genes, n_groups);

    for (batch_idx, batch) in gene_indices.chunks(GENE_PSEUDO_BULK_BATCH).enumerate() {
        let start = Instant::now();
        let chunks = reader.read_gene_parallel(batch)?;
        let row_offset = batch_idx * GENE_PSEUDO_BULK_BATCH;

        let rows: Vec<Vec<f64>> = chunks
            .par_iter()
            .map(|chunk| {
                let mut row = vec![0.0_f64; n_groups];
                match bulk_type {
                    PseudoBulk::Raw => {
                        for (value, &cell) in chunk.data_raw.iter().zip(chunk.indices.iter()) {
                            let c = cell as usize;
                            for g in &membership[offsets[c] as usize..offsets[c + 1] as usize] {
                                row[*g as usize] += value as f64;
                            }
                        }
                    }
                    PseudoBulk::Norm => {
                        // Widen out of f16 before accumulating; summing in f16
                        // biases upwards once the running total passes 64.
                        for (value, &cell) in chunk.data_norm.iter().zip(chunk.indices.iter()) {
                            let v = value.to_f32() as f64;
                            let c = cell as usize;
                            for g in &membership[offsets[c] as usize..offsets[c + 1] as usize] {
                                row[*g as usize] += v;
                            }
                        }
                        for (r, size) in row.iter_mut().zip(sizes.iter()) {
                            *r /= size;
                        }
                    }
                }
                row
            })
            .collect();

        for (i, row) in rows.into_iter().enumerate() {
            for (g, v) in row.into_iter().enumerate() {
                out[(row_offset + i, g)] = v;
            }
        }

        if verbosity.normal_verbosity() {
            println!(
                "Pseudo-bulked genes {} to {} of {} (took {:.2?})",
                row_offset + 1,
                row_offset + batch.len(),
                n_genes,
                start.elapsed()
            );
        }
    }

    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::single_cell::sc_data::in_memory_io::InMemorySparseReader;

    //////////////////////
    // Index remapping //
    //////////////////////

    /// Cell ids at or beyond `n_cells` are dropped without a word. Pinned so a
    /// caller passing the wrong `n_cells` cannot start silently losing cells.
    #[test]
    fn metacells_to_assignments_drops_out_of_range_cell_ids() {
        let mc0: &[usize] = &[0, 5];
        let mc1: &[usize] = &[2, 99];
        let assignments = metacells_to_assignments(&[mc0, mc1], 3);

        assert_eq!(assignments, vec![Some(0), None, Some(1)]);
    }

    /// On overlap the last metacell in the slice wins. The function has no
    /// disjointness check, so the ordering is the whole contract.
    #[test]
    fn metacells_to_assignments_lets_later_metacells_overwrite() {
        let mc0: &[usize] = &[0, 1];
        let mc1: &[usize] = &[1, 2];
        let assignments = metacells_to_assignments(&[mc0, mc1], 3);

        assert_eq!(assignments, vec![Some(0), Some(1), Some(1)]);
    }

    /// Cells outside the subset stay `None`, and the mapping is by position in
    /// `subset_to_orig`, not by value.
    #[test]
    fn remap_assignments_to_original_scatters_by_subset_position() {
        let subset = vec![Some(0), None, Some(1)];
        let subset_to_orig = [4, 2, 0];
        let full = remap_assignments_to_original(&subset, &subset_to_orig, 6);

        assert_eq!(full, vec![Some(1), None, None, None, Some(0), None]);
    }

    /// A `subset_assignments` longer than `subset_to_orig` is truncated rather
    /// than panicking, because the lookup goes through `.get`.
    #[test]
    fn remap_assignments_to_original_ignores_trailing_assignments() {
        let subset = vec![Some(0), Some(1), Some(2)];
        let subset_to_orig = [1];
        let full = remap_assignments_to_original(&subset, &subset_to_orig, 3);

        assert_eq!(full, vec![None, Some(0), None]);
    }

    /// Straight index translation, group structure untouched.
    #[test]
    fn remap_metacells_to_original_translates_every_index() {
        let mc0: &[usize] = &[0, 2];
        let mc1: &[usize] = &[1];
        let remapped = remap_metacells_to_original(&[mc0, mc1], &[10, 20, 30]);

        assert_eq!(remapped, vec![vec![10, 30], vec![20]]);
    }

    /// Defect pin: unlike its sibling this one indexes `subset_to_orig`
    /// directly, so a stale subset index aborts the process instead of
    /// returning an error.
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn remap_metacells_to_original_panics_on_a_stale_index() {
        let mc0: &[usize] = &[0, 5];
        let _ = remap_metacells_to_original(&[mc0], &[10, 20]);
    }

    //////////////////////
    // Pseudo-bulk enum //
    //////////////////////

    /// The string aliases are the R-facing contract and break silently on a
    /// rename, including the British/American spelling pair.
    #[test]
    fn parse_pseudo_bulk_accepts_both_spellings_and_any_case() {
        assert!(matches!(parse_pseudo_bulk("raw"), Some(PseudoBulk::Raw)));
        assert!(matches!(parse_pseudo_bulk("RAW"), Some(PseudoBulk::Raw)));
        assert!(matches!(parse_pseudo_bulk("norm"), Some(PseudoBulk::Norm)));
        assert!(matches!(
            parse_pseudo_bulk("normalised"),
            Some(PseudoBulk::Norm)
        ));
        assert!(matches!(
            parse_pseudo_bulk("Normalized"),
            Some(PseudoBulk::Norm)
        ));
    }

    /// Anything unrecognised falls back to `None` rather than defaulting to
    /// raw counts.
    #[test]
    fn parse_pseudo_bulk_rejects_unknown_labels() {
        assert!(parse_pseudo_bulk("").is_none());
        assert!(parse_pseudo_bulk("counts").is_none());
        assert!(parse_pseudo_bulk("normalise").is_none());
    }

    /// The derived `Default` is what an omitted R argument lands on.
    #[test]
    fn pseudo_bulk_defaults_to_raw() {
        assert!(matches!(PseudoBulk::default(), PseudoBulk::Raw));
    }

    // -- pseudo_bulk_genes_dense --

    /// A dense 6-cell by 4-gene block as CSC, with the normalised layer set to
    /// ten times the raw counts so the two paths are told apart by more than
    /// rounding.
    ///
    /// ```text
    /// gene\cell   0    1    2    3    4    5
    ///    0        1    2    0    4    0    6
    ///    1        0    0    3    0    5    0
    ///    2        2    2    2    2    2    2
    ///    3        0    0    0    0    0    0
    /// ```
    fn tiny_csc() -> CompressedSparseData2<u32, f32> {
        let raw: Vec<u32> = vec![1, 2, 4, 6, 3, 5, 2, 2, 2, 2, 2, 2];
        let norm: Vec<f32> = raw.iter().map(|v| *v as f32 * 10.0).collect();
        // CSC over genes: gene 0 has cells 0,1,3,5; gene 1 has 2,4; gene 2 has
        // all six; gene 3 is empty.
        let indices: Vec<u32> = vec![0, 1, 3, 5, 2, 4, 0, 1, 2, 3, 4, 5];
        let indptr: Vec<u32> = vec![0, 4, 6, 12, 12];
        CompressedSparseData2::new_csc(&raw, &indices, &indptr, Some(&norm), (6, 4))
    }

    /// Raw pseudo-bulk sums counts within each group.
    #[test]
    fn test_pseudo_bulk_genes_dense_raw_sums() {
        let matrix = tiny_csc();
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let groups = vec![vec![0usize, 1, 2], vec![3usize, 4, 5]];

        let out =
            pseudo_bulk_genes_dense(&reader, &[0, 1, 2, 3], &groups, PseudoBulk::Raw, 0).unwrap();

        assert_eq!(out.nrows(), 4);
        assert_eq!(out.ncols(), 2);
        // gene 0: 1 + 2 + 0 = 3, then 4 + 0 + 6 = 10
        assert_eq!(out[(0, 0)], 3.0);
        assert_eq!(out[(0, 1)], 10.0);
        // gene 1: 0 + 0 + 3 = 3, then 0 + 5 + 0 = 5
        assert_eq!(out[(1, 0)], 3.0);
        assert_eq!(out[(1, 1)], 5.0);
        // gene 2 is uniform, gene 3 is empty
        assert_eq!(out[(2, 0)], 6.0);
        assert_eq!(out[(2, 1)], 6.0);
        assert_eq!(out[(3, 0)], 0.0);
        assert_eq!(out[(3, 1)], 0.0);
    }

    /// Normalised pseudo-bulk averages over every cell in the group, implicit
    /// zeros included. That is the `colMeans` of a dense block, not the mean of
    /// the stored values.
    #[test]
    fn test_pseudo_bulk_genes_dense_norm_averages_over_zeros() {
        let matrix = tiny_csc();
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let groups = vec![vec![0usize, 1, 2], vec![3usize, 4, 5]];

        let out =
            pseudo_bulk_genes_dense(&reader, &[0, 1, 2, 3], &groups, PseudoBulk::Norm, 0).unwrap();

        // gene 0 group 0: (10 + 20 + 0) / 3, not / 2.
        assert!((out[(0, 0)] - 10.0).abs() < 1e-6);
        assert!((out[(0, 1)] - 100.0 / 3.0).abs() < 1e-6);
        assert!((out[(1, 0)] - 10.0).abs() < 1e-6);
        assert!((out[(2, 0)] - 20.0).abs() < 1e-6);
        assert_eq!(out[(3, 0)], 0.0);
    }

    /// Rows follow the caller's gene order, not the store's.
    #[test]
    fn test_pseudo_bulk_genes_dense_honours_gene_order() {
        let matrix = tiny_csc();
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let groups = vec![vec![0usize, 1, 2], vec![3usize, 4, 5]];

        let out = pseudo_bulk_genes_dense(&reader, &[2, 0], &groups, PseudoBulk::Raw, 0).unwrap();

        assert_eq!(out.nrows(), 2);
        assert_eq!(out[(0, 0)], 6.0); // gene 2 first
        assert_eq!(out[(1, 0)], 3.0); // then gene 0
    }

    /// Overlapping groups are allowed; a shared cell counts in both.
    #[test]
    fn test_pseudo_bulk_genes_dense_allows_overlapping_groups() {
        let matrix = tiny_csc();
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let groups = vec![vec![0usize, 1], vec![1usize, 3]];

        let out = pseudo_bulk_genes_dense(&reader, &[0], &groups, PseudoBulk::Raw, 0).unwrap();

        assert_eq!(out[(0, 0)], 3.0); // cells 0 and 1 -> 1 + 2
        assert_eq!(out[(0, 1)], 6.0); // cells 1 and 3 -> 2 + 4
    }

    #[test]
    fn test_pseudo_bulk_genes_dense_rejects_bad_groups() {
        let matrix = tiny_csc();
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        // Cell index past the end of the store.
        let bad = vec![vec![0usize, 99]];
        assert!(pseudo_bulk_genes_dense(&reader, &[0], &bad, PseudoBulk::Raw, 0).is_err());
        // An empty group has no denominator.
        let empty = vec![vec![0usize], Vec::new()];
        assert!(pseudo_bulk_genes_dense(&reader, &[0], &empty, PseudoBulk::Raw, 0).is_err());
    }
}
