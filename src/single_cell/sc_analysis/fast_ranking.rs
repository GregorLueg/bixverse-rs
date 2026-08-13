//! Helpers for fast ranking of expression for differential gene expression or
//! or AUCell type analyses.

use rayon::prelude::*;

use crate::prelude::*;

//////////////////
// Single cells //
//////////////////

/// Helper function to rank specifically `F16` type slices
///
/// ### Params
///
/// * `vec` - Slice of `F16`
///
/// ### Returns
///
/// The ranked values as an f32 vector.
fn rank_f16(vec: &[F16]) -> Vec<f32> {
    let n = vec.len();
    if n == 0 {
        return Vec::new();
    }

    // F16 bit pattern is monotonic in value for non-negative finite values
    // (IEEE 754 sign-magnitude). Normalised counts are >= 0, so sorting by
    // raw u16 bits matches sorting by value, with a cheaper integer compare.
    let mut indexed: Vec<(u16, usize)> = vec
        .iter()
        .enumerate()
        .map(|(i, v)| (v.to_bits(), i))
        .collect();

    indexed.sort_unstable_by_key(|&(bits, _)| bits);

    let mut ranks: Vec<f32> = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let current = indexed[i].0;
        let start = i;
        while i < n && indexed[i].0 == current {
            i += 1;
        }
        let avg_rank = (start + i + 1) as f32 / 2.0;
        for j in start..i {
            ranks[indexed[j].1] = avg_rank;
        }
    }
    ranks
}

/// Fast ranking of CSR-type data for single cell
///
/// The function takes in CSR-style data (rows = cells, columns = genes) and
/// generates ranked versions of the data.
///
/// ### Params
///
/// * `row_ptr` - The row pointer in the given CSR data
/// * `col_indices` - The col indices of the data
/// * `data` - The normalised count data which to rank
/// * `nrow` - Number of rows (cells)
/// * `ncol` - Number of columns (genes)
/// * `rank_within_rows` - This boolean controls if the ranking happens within
///   cells (for example for AUCell) or across genes (for example for DGE).
///
/// ### Return
///
/// A `Vec<Vec<f32>>` that pending the rank_within_rows represents the ranks
/// across genes or across cells.
pub fn fast_csr_ranking(
    row_ptr: &[usize],
    col_indices: &[u32],
    data: &[F16],
    nrow: usize,
    ncol: usize,
    rank_within_rows: bool,
) -> Vec<Vec<f32>> {
    if rank_within_rows {
        // Rank genes within each cell
        // This is what we are interested in for AUCell type approaches
        (0..nrow)
            .into_par_iter()
            .map(|row_idx| {
                let start = row_ptr[row_idx];
                let end = row_ptr[row_idx + 1];
                let num_nonzeros = end - start;
                let num_zeros = ncol - num_nonzeros;

                if num_nonzeros == 0 {
                    let zero_rank = (1.0 + ncol as f32) / 2.0;
                    return vec![zero_rank; ncol];
                }

                if num_zeros == 0 {
                    let row_data = &data[start..end];
                    return rank_f16(row_data);
                }

                let row_data = &data[start..end];
                let row_cols = &col_indices[start..end];
                let nonzero_ranks = rank_f16(row_data);
                let zero_rank = (1.0 + num_zeros as f32) / 2.0;
                let mut result = vec![zero_rank; ncol];

                for (i, &col) in row_cols.iter().enumerate() {
                    result[col as usize] = nonzero_ranks[i] + num_zeros as f32;
                }

                result
            })
            .collect()
    } else {
        // Rank cells within each gene - build gene-to-cells mapping first
        let mut gene_data: Vec<Vec<(u16, usize)>> = vec![Vec::new(); ncol];

        // Single pass: collect all data per gene (store raw bits for fast sort)
        for row_idx in 0..nrow {
            let start = row_ptr[row_idx];
            let end = row_ptr[row_idx + 1];

            for i in 0..(end - start) {
                let col_idx = col_indices[start + i] as usize;
                gene_data[col_idx].push((data[start + i].to_bits(), row_idx));
            }
        }

        // Rank each gene in parallel
        gene_data
            .into_par_iter()
            .map(|mut values| {
                let num_nonzeros = values.len();
                let num_zeros = nrow - num_nonzeros;

                if num_nonzeros == 0 {
                    let zero_rank = (1.0 + nrow as f32) / 2.0;
                    return vec![zero_rank; nrow];
                }

                values.sort_unstable_by_key(|&(bits, _)| bits);

                let zero_rank = (1.0 + num_zeros as f32) / 2.0;
                let mut result = vec![zero_rank; nrow];

                let mut i = 0;
                while i < num_nonzeros {
                    let start_idx = i;
                    let current_value = values[i].0;
                    while i < num_nonzeros && values[i].0 == current_value {
                        i += 1;
                    }
                    let avg_rank = (start_idx + i + 1 + 2 * num_zeros) as f32 / 2.0;
                    for j in start_idx..i {
                        result[values[j].1] = avg_rank;
                    }
                }

                result
            })
            .collect()
    }
}

/// Tie-correction contribution of a single tie group.
///
/// `t^3 - t`, which is zero for `t <= 1`, so the caller never needs to branch.
///
/// ### Params
///
/// * `t` - Size of the tie group.
///
/// ### Returns
///
/// The `t^3 - t` term for the Mann-Whitney variance correction.
#[inline(always)]
fn tie_contribution(t: usize) -> f64 {
    let t = t as f64;
    t * t * t - t
}

/// Per-gene rank-sum statistics for two groups of cells, fused into the scan
///
/// Group 1 occupies rows `0..n_grp1` of the CSR data and group 2 the remainder,
/// so the caller concatenates the two groups in that order. Computes the same
/// midranks as [fast_csr_ranking] with `rank_within_rows = false`, but reduces
/// them to a rank sum and a tie term inside the block walk instead of
/// materialising the `n_genes x n_cells` rank matrix. Peak memory is therefore
/// `O(nnz)` rather than `O(ncol * nrow)`, which is the difference between a
/// few hundred MB and several GB on a realistic comparison.
///
/// Both accumulators are `f64`. The rank sum reaches ~2.5e9 at 50k cells,
/// well past what `f32` can accumulate without swamping the test statistic.
///
/// ### Params
///
/// * `row_ptr` - The row pointer in the given CSR data.
/// * `col_indices` - The col indices of the data.
/// * `data` - The normalised count data.
/// * `n_grp1` - Number of leading rows belonging to group 1.
/// * `nrow` - Number of rows (cells) across both groups.
/// * `ncol` - Number of columns (genes).
///
/// ### Returns
///
/// One `(rank_sum_grp1, tie_term)` per gene, where `tie_term` is `sum(t^3 - t)`
/// over the gene's tie groups, including the block of implicit zeros.
pub fn csr_rank_sum_stats_two_groups(
    row_ptr: &[usize],
    col_indices: &[u32],
    data: &[F16],
    n_grp1: usize,
    nrow: usize,
    ncol: usize,
) -> Vec<(f64, f64)> {
    // (u16, u32) is 8 bytes against 16 for (u16, usize) after alignment
    // padding, and this buffer is the dominant transient allocation.
    let mut gene_data: Vec<Vec<(u16, u32)>> = vec![Vec::new(); ncol];

    for row_idx in 0..nrow {
        let start = row_ptr[row_idx];
        let end = row_ptr[row_idx + 1];

        for i in start..end {
            gene_data[col_indices[i] as usize].push((data[i].to_bits(), row_idx as u32));
        }
    }

    gene_data
        .into_par_iter()
        .map(|mut values| {
            let num_nonzeros = values.len();
            let num_zeros = nrow - num_nonzeros;

            let mut rank_sum = 0.0_f64;
            let mut tie_term = tie_contribution(num_zeros);
            let mut nonzeros_grp1 = 0_usize;

            if num_nonzeros > 0 {
                values.sort_unstable_by_key(|&(bits, _)| bits);

                let mut i = 0;
                while i < num_nonzeros {
                    let start_idx = i;
                    let current_value = values[i].0;
                    let mut in_grp1 = 0_usize;
                    while i < num_nonzeros && values[i].0 == current_value {
                        if (values[i].1 as usize) < n_grp1 {
                            in_grp1 += 1;
                        }
                        i += 1;
                    }
                    let midrank = (start_idx + i + 1 + 2 * num_zeros) as f64 / 2.0;
                    rank_sum += in_grp1 as f64 * midrank;
                    tie_term += tie_contribution(i - start_idx);
                    nonzeros_grp1 += in_grp1;
                }
            }

            // Whatever is left of group 1 sits in the shared zero block
            let zeros_grp1 = n_grp1 - nonzeros_grp1;
            rank_sum += zeros_grp1 as f64 * (1.0 + num_zeros as f64) / 2.0;

            (rank_sum, tie_term)
        })
        .collect()
}

/// Append a group of cells to flat CSR buffers
///
/// Lets a caller build the CSR of one group once and then swap the second
/// group in via `truncate` plus another append, rather than re-flattening both
/// groups for every comparison.
///
/// ### Params
///
/// * `chunks` - The cells to append, one CSR row each.
/// * `indptr` - Row pointer, which the caller seeds with a single `0`.
/// * `indices` - Column indices, appended to.
/// * `data` - Normalised counts, appended to.
pub(crate) fn append_cell_chunks(
    chunks: &[CsrCellChunk],
    indptr: &mut Vec<usize>,
    indices: &mut Vec<u32>,
    data: &mut Vec<F16>,
) {
    let mut current = *indptr.last().unwrap_or(&0);

    for chunk in chunks {
        data.extend_from_slice(&chunk.data_norm);
        indices.extend_from_slice(&chunk.indices);
        current += chunk.data_norm.len();
        indptr.push(current);
    }
}

/// Helper function to rank all cells within a given chunk vector
///
/// ### Params
///
/// * `chunk_vec` - Vector of `CsrCellChunk` to rank.
/// * `no_genes` - Number of represented genes in this data.
/// * `rank_within_rows` - This boolean controls if the ranking happens within
///   cells (for example for AUCell) or across genes (for example for DGE).
///
/// ### Returns
///
/// A `Vec<Vec<f32>>` that pending the rank_within_rows represents the ranks
/// across genes or across cells.
pub fn rank_csr_chunk_vec(
    chunk_vec: Vec<CsrCellChunk>,
    no_genes: usize,
    rank_within_rows: bool,
) -> Vec<Vec<f32>> {
    let no_cells = chunk_vec.len();
    let mut all_data: Vec<Vec<F16>> = Vec::with_capacity(chunk_vec.len());
    let mut all_indices: Vec<Vec<u32>> = Vec::with_capacity(chunk_vec.len());
    let mut indptr: Vec<usize> = Vec::with_capacity(chunk_vec.len() + 1);
    let mut current_indptr = 0_usize;

    indptr.push(current_indptr);

    for chunk in chunk_vec {
        let data_len = chunk.data_norm.len();
        all_data.push(chunk.data_norm);
        all_indices.push(chunk.indices);
        current_indptr += data_len;
        indptr.push(current_indptr);
    }

    let all_data = flatten_vector(all_data);
    let all_indices = flatten_vector(all_indices);

    fast_csr_ranking(
        &indptr,
        &all_indices,
        &all_data,
        no_cells,
        no_genes,
        rank_within_rows,
    )
}

///////////////
// MetaCells //
///////////////

/// Rank an f32 slice with average ranks for ties.
///
/// ### Params
///
/// * `vec` - Slice of `f32`
///
/// ### Returns
///
/// The ranked values as an f32 vector.
pub fn rank_f32(vec: &[f32]) -> Vec<f32> {
    let n = vec.len();
    if n == 0 {
        return Vec::new();
    }

    let mut indexed: Vec<(f32, usize)> = vec
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();

    indexed.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0_f32; n];
    let mut i = 0;
    while i < n {
        let current = indexed[i].0;
        let start = i;
        while i < n && indexed[i].0 == current {
            i += 1;
        }
        let avg_rank = (start + i + 1) as f32 / 2.0;
        for j in start..i {
            ranks[indexed[j].1] = avg_rank;
        }
    }
    ranks
}

/// Rank genes within each cell for a CSR (cells x genes) layout, f32 data.
///
/// ### Params
///
/// * `indptr` - The row indices
/// * `indices` - The column indices
/// * `data` - The underlying normalised data
/// * `n_cells` - Number of cells
/// * `n_genes` - Number of genes
pub fn rank_within_rows_f32(
    indptr: &[usize],
    indices: &[usize],
    data: &[f32],
    n_cells: usize,
    n_genes: usize,
) -> Vec<Vec<f32>> {
    (0..n_cells)
        .into_par_iter()
        .map(|row_idx| {
            let start = indptr[row_idx];
            let end = indptr[row_idx + 1];
            let num_nonzeros = end - start;
            let num_zeros = n_genes - num_nonzeros;

            if num_nonzeros == 0 {
                let zero_rank = (1.0 + n_genes as f32) / 2.0;
                return vec![zero_rank; n_genes];
            }

            if num_zeros == 0 {
                return rank_f32(&data[start..end]);
            }

            let nonzero_ranks = rank_f32(&data[start..end]);
            let zero_rank = (1.0 + num_zeros as f32) / 2.0;
            let mut result = vec![zero_rank; n_genes];

            for (i, &col) in indices[start..end].iter().enumerate() {
                result[col] = nonzero_ranks[i] + num_zeros as f32;
            }

            result
        })
        .collect()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::single_cell::sc_traits::F16;
    use approx::assert_relative_eq;

    // Helper to create F16 from f32
    fn f16_vec(values: &[f32]) -> Vec<F16> {
        values.iter().map(|&v| F16::from_f32(v)).collect()
    }

    /// A row with no stored entries ranks as one tie across the full gene width.
    #[test]
    fn test_all_zeros_row() {
        let row_ptr = vec![0, 0, 3];
        let col_indices = vec![0, 1, 2];
        let data = f16_vec(&[1.0, 2.0, 3.0]);

        let result = fast_csr_ranking(&row_ptr, &col_indices, &data, 2, 3, true);

        assert_eq!(result[0], vec![2.0, 2.0, 2.0]);
        assert_eq!(result[1], vec![1.0, 2.0, 3.0]);
    }

    /// Implicit zeros share the average of the ranks they span; stored values rank above.
    #[test]
    fn test_multiple_tied_zeros() {
        let row_ptr = vec![0, 2];
        let col_indices = vec![0, 3];
        let data = f16_vec(&[1.0, 2.0]);

        let result = fast_csr_ranking(&row_ptr, &col_indices, &data, 1, 4, true);

        let expected = [3.0, 1.5, 1.5, 4.0];
        let actual = &result[0];

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 0.01, "Expected {}, got {}", e, a);
        }
    }

    /// `rank_within_rows` switches between ranking genes inside a cell and cells inside a gene.
    #[test]
    fn test_row_vs_column_ranking() {
        let row_ptr = vec![0, 3, 4];
        let col_indices = vec![0, 1, 2, 0];
        let data = f16_vec(&[1.0, 2.0, 5.0, 3.0]);

        let row_result = fast_csr_ranking(&row_ptr, &col_indices, &data, 2, 3, true);
        let col_result = fast_csr_ranking(&row_ptr, &col_indices, &data, 2, 3, false);

        assert_eq!(row_result.len(), 2);
        assert_eq!(row_result[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(row_result[1], vec![3.0, 1.5, 1.5]);

        assert_eq!(col_result.len(), 3);
        assert_eq!(col_result[0], vec![1.0, 2.0]);
        assert_eq!(col_result[1], vec![2.0, 1.0]);
        assert_eq!(col_result[2], vec![2.0, 1.0]);
    }

    /// Ties between stored values within a column also get the averaged rank.
    #[test]
    fn test_column_ranking_with_ties() {
        let row_ptr = vec![0, 2, 3, 4];
        let col_indices = vec![0, 1, 1, 0];
        let data = f16_vec(&[2.0, 1.0, 1.0, 2.0]);

        let result = fast_csr_ranking(&row_ptr, &col_indices, &data, 3, 2, false);

        let gene0_actual = &result[0];
        let gene1_actual = &result[1];

        assert!((gene0_actual[0] - 2.5).abs() < 0.01);
        assert!((gene0_actual[1] - 1.0).abs() < 0.01);
        assert!((gene0_actual[2] - 2.5).abs() < 0.01);

        assert!((gene1_actual[0] - 2.5).abs() < 0.01);
        assert!((gene1_actual[1] - 2.5).abs() < 0.01);
        assert!((gene1_actual[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rank_sum_stats_matches_materialised() {
        // Anchor test: the fused kernel must agree with summing the group 1
        // slice of the already-tested materialised ranking.
        // Matrix (6 cells x 4 genes), first 3 cells are group 1:
        // [2.0, 0.0, 1.0, 0.0]
        // [0.0, 3.0, 1.0, 0.0]
        // [5.0, 0.0, 0.0, 0.0]
        // [1.0, 3.0, 4.0, 0.0]
        // [0.0, 0.0, 1.0, 0.0]
        // [2.0, 1.0, 0.0, 0.0]
        let row_ptr = vec![0, 2, 4, 5, 8, 9, 11];
        let col_indices: Vec<u32> = vec![0, 2, 1, 2, 0, 0, 1, 2, 2, 0, 1];
        let data = f16_vec(&[2.0, 1.0, 3.0, 1.0, 5.0, 1.0, 3.0, 4.0, 1.0, 2.0, 1.0]);

        let n_grp1 = 3;
        let (nrow, ncol) = (6, 4);

        let ranks = fast_csr_ranking(&row_ptr, &col_indices, &data, nrow, ncol, false);
        let stats =
            csr_rank_sum_stats_two_groups(&row_ptr, &col_indices, &data, n_grp1, nrow, ncol);

        for gene in 0..ncol {
            let expected: f64 = ranks[gene][..n_grp1].iter().map(|&r| r as f64).sum();
            assert_relative_eq!(stats[gene].0, expected, epsilon = 1e-9);
        }

        // Gene 3 is empty, so all six cells share one tie group
        assert_relative_eq!(stats[3].0, 3.0 * 3.5, epsilon = 1e-9);
        assert_relative_eq!(stats[3].1, 6.0 * 6.0 * 6.0 - 6.0, epsilon = 1e-9);
    }

    #[test]
    fn test_rank_sum_stats_tie_term() {
        // Single gene over 6 cells: values [1.0, 1.0, 1.0, 2.0, 2.0, 0.0].
        // Tie groups: three 1.0s, two 2.0s and a single implicit zero.
        // S = (27 - 3) + (8 - 2) + (1 - 1) = 30
        let row_ptr = vec![0, 1, 2, 3, 4, 5, 5];
        let col_indices: Vec<u32> = vec![0, 0, 0, 0, 0];
        let data = f16_vec(&[1.0, 1.0, 1.0, 2.0, 2.0]);

        let stats = csr_rank_sum_stats_two_groups(&row_ptr, &col_indices, &data, 3, 6, 1);

        assert_relative_eq!(stats[0].1, 30.0, epsilon = 1e-9);
        // Zero sits at rank 1, the three 1.0s share midrank 3, the two 2.0s
        // share midrank 5.5. Group 1 is the first three cells, all 1.0s.
        assert_relative_eq!(stats[0].0, 9.0, epsilon = 1e-9);
    }

    #[test]
    fn test_rank_sum_stats_no_ties() {
        // Six distinct values, no zeros: S must be exactly 0.
        let row_ptr = vec![0, 1, 2, 3, 4, 5, 6];
        let col_indices: Vec<u32> = vec![0, 0, 0, 0, 0, 0];
        let data = f16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let stats = csr_rank_sum_stats_two_groups(&row_ptr, &col_indices, &data, 3, 6, 1);

        assert_relative_eq!(stats[0].1, 0.0, epsilon = 1e-9);
        // Group 1 holds the three lowest values, so ranks 1 + 2 + 3
        assert_relative_eq!(stats[0].0, 6.0, epsilon = 1e-9);
    }

    #[test]
    fn test_append_cell_chunks_round_trip() {
        let chunks = [
            CsrCellChunk::from_data(&[1_u32, 3], &[0_u32, 2], 0, 1e4, true),
            CsrCellChunk::from_data(&[2_u32], &[1_u32], 1, 1e4, true),
        ];

        let mut indptr = vec![0_usize];
        let mut indices: Vec<u32> = Vec::new();
        let mut data: Vec<F16> = Vec::new();

        append_cell_chunks(&chunks[..1], &mut indptr, &mut indices, &mut data);
        let prefix_rows = indptr.len();
        let prefix_nnz = indices.len();

        append_cell_chunks(&chunks[1..], &mut indptr, &mut indices, &mut data);
        assert_eq!(indptr, vec![0, 2, 3]);
        assert_eq!(indices, vec![0, 2, 1]);

        // Truncating back to the prefix must restore the first append exactly
        indptr.truncate(prefix_rows);
        indices.truncate(prefix_nnz);
        data.truncate(prefix_nnz);
        assert_eq!(indptr, vec![0, 2]);
        assert_eq!(indices, vec![0, 2]);
    }
}
