//! Helpers for fast ranking of expression for differential gene expression or
//! or AUCell type analyses.

use rayon::prelude::*;

use crate::prelude::*;

////////////
// Consts //
////////////

/// Group label of a cell that sits in no group and is skipped by the scan.
pub const NO_GROUP: u16 = u16::MAX;

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

/// Rank one cell's expression into a reused dense buffer
///
/// Same ascending midranks as the `rank_within_rows` branch of
/// [fast_csr_ranking]: implicit zeros share one midrank and the stored values
/// are ranked above them.
///
/// ### Params
///
/// * `indices` - Gene indices of the stored values
/// * `data` - Stored (normalised) values, same length as `indices`
/// * `n_genes` - Total number of genes
/// * `out` - Reused output buffer, resized to `n_genes`
/// * `sort_buf` - Reused sort scratch
///
/// ### Returns
///
/// Nothing; `out` holds one midrank per gene.
pub(crate) fn rank_cell_into(
    indices: &[u32],
    data: &[F16],
    n_genes: usize,
    out: &mut Vec<f32>,
    sort_buf: &mut Vec<(u16, usize)>,
) {
    let num_nonzeros = data.len();
    let num_zeros = n_genes - num_nonzeros;
    out.clear();

    if num_nonzeros == 0 {
        out.resize(n_genes, (1.0 + n_genes as f32) / 2.0);
        return;
    }

    let (zero_shift, target): (f32, &dyn Fn(usize) -> usize) = if num_zeros == 0 {
        out.resize(n_genes, 0.0);
        (0.0, &|i| i)
    } else {
        out.resize(n_genes, (1.0 + num_zeros as f32) / 2.0);
        (num_zeros as f32, &|i| indices[i] as usize)
    };

    sort_buf.clear();
    sort_buf.extend(data.iter().enumerate().map(|(i, v)| (v.to_bits(), i)));
    sort_buf.sort_unstable_by_key(|&(bits, _)| bits);

    let n = sort_buf.len();
    let mut i = 0;
    while i < n {
        let current = sort_buf[i].0;
        let start = i;
        while i < n && sort_buf[i].0 == current {
            i += 1;
        }
        let avg_rank = (start + i + 1) as f32 / 2.0;
        for &(_, pos) in &sort_buf[start..i] {
            out[target(pos)] = avg_rank + zero_shift;
        }
    }
}

/// Bucket the stored values of a CSR matrix by column in a single flat buffer
///
/// A counting pass sizes every column exactly, so the fill is one sequential
/// pass over the non-zeros with no reallocation. Within a column the entries
/// keep row order.
///
/// ### Params
///
/// * `row_ptr` - CSR row pointers
/// * `col_indices` - CSR column indices
/// * `nrow` - Number of rows
/// * `ncol` - Number of columns
/// * `make` - Builds the stored entry from `(position in the CSR arrays, row)`
///
/// ### Returns
///
/// `(flat, offsets)`, where column `j` is `flat[offsets[j]..offsets[j + 1]]`
fn bucket_by_column<V: Copy + Default>(
    row_ptr: &[usize],
    col_indices: &[u32],
    nrow: usize,
    ncol: usize,
    make: impl Fn(usize, usize) -> V,
) -> (Vec<V>, Vec<usize>) {
    let nnz = row_ptr[nrow];
    let mut offsets = vec![0_usize; ncol + 1];
    for &col in &col_indices[..nnz] {
        offsets[col as usize + 1] += 1;
    }
    for j in 0..ncol {
        offsets[j + 1] += offsets[j];
    }

    let mut cursor = offsets[..ncol].to_vec();
    let mut flat = vec![V::default(); nnz];
    for row in 0..nrow {
        for i in row_ptr[row]..row_ptr[row + 1] {
            let col = col_indices[i] as usize;
            flat[cursor[col]] = make(i, row);
            cursor[col] += 1;
        }
    }

    (flat, offsets)
}

/// Split a flat buffer into disjoint mutable per-column slices
///
/// ### Params
///
/// * `flat` - The flat buffer
/// * `offsets` - Column offsets from [bucket_by_column]
///
/// ### Returns
///
/// One mutable slice per column.
fn split_by_offsets<'a, V>(flat: &'a mut [V], offsets: &[usize]) -> Vec<&'a mut [V]> {
    let mut rest = flat;
    let mut out = Vec::with_capacity(offsets.len() - 1);
    for w in offsets.windows(2) {
        let (head, tail) = rest.split_at_mut(w[1] - w[0]);
        out.push(head);
        rest = tail;
    }
    out
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
        // Counting pass sizes every gene, then one fill pass (raw bits for the sort)
        let (mut flat, offsets) = bucket_by_column(row_ptr, col_indices, nrow, ncol, |i, row| {
            (data[i].to_bits(), row)
        });

        // Rank each gene in parallel
        split_by_offsets(&mut flat, &offsets)
            .into_par_iter()
            .map(|values| {
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
    let (mut flat, offsets) = bucket_by_column(row_ptr, col_indices, nrow, ncol, |i, row| {
        (data[i].to_bits(), row as u32)
    });

    split_by_offsets(&mut flat, &offsets)
        .into_par_iter()
        .map(|values| {
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

/// `t^3 - t` in exact integer arithmetic.
///
/// `u128` because the implicit-zero block alone reaches `t^3 ~ 1e19` at two
/// million cells, past `u64`.
///
/// ### Params
///
/// * `t` - Size of the tie group.
///
/// ### Returns
///
/// The `t^3 - t` term.
#[inline(always)]
fn tie_contribution_exact(t: u64) -> u128 {
    let t = t as u128;
    t * t * t - t
}

////////////////////
// GeneGroupStats //
////////////////////

/// Per-gene Mann-Whitney statistics for every pair of cell groups, from one
/// sort of the gene's non-zeros.
///
/// U is additive over disjoint groups, so the one-vs-rest statistic of group
/// `g` is the row sum of `U(g, .)` against the tie structure of all grouped
/// cells pooled, which the sweep tracks as well.
///
/// The struct doubles as per-thread scratch: [Self::gather] and [Self::sweep]
/// reset what they write, so one instance serves every gene a thread sees.
pub struct GeneGroupStats {
    /// Number of groups
    n_groups: usize,
    /// Non-zero cells per group
    pub nnz: Vec<u64>,
    /// Sum of the normalised values per group
    pub sum: Vec<f64>,
    /// `2 U(g, h)`, row-major `G x G`, i.e. twice the number of `(g, h)` cell
    /// pairs where the `g` cell is larger, ties counting a half. The diagonal
    /// is meaningless.
    u2: Vec<u64>,
    /// `sum (e_g^3 - e_g)` over the gene's tie blocks, per group
    tie_own: Vec<u128>,
    /// `sum e_g^2 e_h` over the gene's tie blocks, row-major `G x G`
    tie_cross: Vec<u128>,
    /// `sum (t^3 - t)` over tie blocks of all grouped cells pooled
    tie_pooled: u128,
    /// The gene's grouped non-zeros as `(f16 bits, group)`
    values: Vec<(u16, u16)>,
    /// Cells per group strictly below the current block
    less_than: Vec<u64>,
    /// Cells per group inside the current block
    equal: Vec<u64>,
    /// Groups present in the current block
    active: Vec<usize>,
}

impl GeneGroupStats {
    /// Allocate the statistics and scratch for `n_groups` groups.
    ///
    /// ### Params
    ///
    /// * `n_groups` - Number of cell groups.
    ///
    /// ### Returns
    ///
    /// The zeroed structure.
    pub fn new(n_groups: usize) -> Self {
        Self {
            n_groups,
            nnz: vec![0; n_groups],
            sum: vec![0.0; n_groups],
            u2: vec![0; n_groups * n_groups],
            tie_own: vec![0; n_groups],
            tie_cross: vec![0; n_groups * n_groups],
            tie_pooled: 0,
            values: Vec::new(),
            less_than: vec![0; n_groups],
            equal: vec![0; n_groups],
            active: Vec::with_capacity(n_groups),
        }
    }

    /// Collect a gene's grouped non-zeros, their counts and sums per group.
    ///
    /// Cheap relative to [Self::sweep], so the caller can apply the proportion
    /// filter in between and skip the sort for genes it drops.
    ///
    /// ### Params
    ///
    /// * `data_norm` - The gene's normalised values.
    /// * `indices` - The gene's cell indices, aligned with `data_norm`.
    /// * `lookup` - Group label per cell of the store, [NO_GROUP] for cells
    ///   outside every group.
    pub fn gather(&mut self, data_norm: &[F16], indices: &[u32], lookup: &[u16]) {
        self.nnz.fill(0);
        self.sum.fill(0.0);
        self.values.clear();

        for (&cell, &value) in indices.iter().zip(data_norm) {
            let group = lookup[cell as usize];
            if group == NO_GROUP {
                continue;
            }
            self.nnz[group as usize] += 1;
            self.sum[group as usize] += value.to_f32() as f64;
            self.values.push((value.to_bits(), group));
        }
    }

    /// Rank the gathered values and accumulate U and the tie terms.
    ///
    /// ### Params
    ///
    /// * `group_sizes` - Number of cells per group.
    pub fn sweep(&mut self, group_sizes: &[usize]) {
        let g_n = self.n_groups;
        self.u2.fill(0);
        self.tie_own.fill(0);
        self.tie_cross.fill(0);
        self.tie_pooled = 0;
        self.less_than.fill(0);

        // the implicit zeros form the lowest block
        for g in 0..g_n {
            self.equal[g] = group_sizes[g] as u64 - self.nnz[g];
            if self.equal[g] > 0 {
                self.active.push(g);
            }
        }
        self.close_block();

        // f16 bits are monotonic in value for non-negative finite values
        let mut values = std::mem::take(&mut self.values);
        values.sort_unstable_by_key(|&(bits, _)| bits);

        let mut i = 0;
        while i < values.len() {
            let bits = values[i].0;
            while i < values.len() && values[i].0 == bits {
                let g = values[i].1 as usize;
                if self.equal[g] == 0 {
                    self.active.push(g);
                }
                self.equal[g] += 1;
                i += 1;
            }
            self.close_block();
        }

        self.values = values;
    }

    /// Fold the current tie block into the accumulators and advance
    /// `less_than` past it.
    fn close_block(&mut self) {
        let g_n = self.n_groups;
        let mut t = 0_u64;

        for &g in &self.active {
            let e = self.equal[g];
            t += e;

            let row = &mut self.u2[g * g_n..(g + 1) * g_n];
            for ((u, &lt), &eq) in row.iter_mut().zip(&self.less_than).zip(&self.equal) {
                *u += e * (2 * lt + eq);
            }

            self.tie_own[g] += tie_contribution_exact(e);
            let e2 = (e * e) as u128;
            for &h in &self.active {
                if h != g {
                    self.tie_cross[g * g_n + h] += e2 * self.equal[h] as u128;
                }
            }
        }
        self.tie_pooled += tie_contribution_exact(t);

        for &g in &self.active {
            self.less_than[g] += self.equal[g];
            self.equal[g] = 0;
        }
        self.active.clear();
    }

    /// U statistic of group `g` against group `h`.
    ///
    /// ### Params
    ///
    /// * `g` - The first group.
    /// * `h` - The second group, `!= g`.
    ///
    /// ### Returns
    ///
    /// `U(g, h)`, ties counting a half.
    #[inline]
    pub fn u_pair(&self, g: usize, h: usize) -> f64 {
        self.u2[g * self.n_groups + h] as f64 / 2.0
    }

    /// `sum (t^3 - t)` over the tie blocks of groups `g` and `h` pooled.
    ///
    /// ### Params
    ///
    /// * `g` - The first group.
    /// * `h` - The second group, `!= g`.
    ///
    /// ### Returns
    ///
    /// The pair's tie term.
    #[inline]
    pub fn tie_pair(&self, g: usize, h: usize) -> f64 {
        let g_n = self.n_groups;
        let cross = self.tie_cross[g * g_n + h] + self.tie_cross[h * g_n + g];
        (self.tie_own[g] + self.tie_own[h] + 3 * cross) as f64
    }

    /// U statistic of group `g` against every other group pooled.
    ///
    /// ### Params
    ///
    /// * `g` - The group.
    ///
    /// ### Returns
    ///
    /// `U(g, rest)`, ties counting a half.
    #[inline]
    pub fn u_rest(&self, g: usize) -> f64 {
        let g_n = self.n_groups;
        let row = &self.u2[g * g_n..(g + 1) * g_n];
        let total: u64 = row.iter().sum::<u64>() - row[g];
        total as f64 / 2.0
    }

    /// `sum (t^3 - t)` over the tie blocks of all grouped cells.
    ///
    /// ### Returns
    ///
    /// The pooled tie term, shared by every one-vs-rest comparison.
    #[inline]
    pub fn tie_pooled(&self) -> f64 {
        self.tie_pooled as f64
    }
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

    /// The fused rank-sum kernel must agree with summing the group 1 slice of
    /// the materialised ranking, which is pinned separately above.
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

    /// The tie-correction term sums `t^3 - t` over every tie group, including
    /// the implicit group the structural zeros form.
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

    /// With every value distinct there is no tie group, so the correction term
    /// has to be exactly zero rather than merely small.
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

    /// Reference `(U, tie term)` of cells `a` against cells `b` for one gene,
    /// via the two-group kernel on a single-column CSR.
    fn two_group_reference(column: &[f32], a: &[usize], b: &[usize]) -> (f64, f64) {
        let mut row_ptr = vec![0_usize];
        let mut col_indices: Vec<u32> = Vec::new();
        let mut data: Vec<F16> = Vec::new();
        for &cell in a.iter().chain(b) {
            if column[cell] != 0.0 {
                col_indices.push(0);
                data.push(F16::from_f32(column[cell]));
            }
            row_ptr.push(col_indices.len());
        }

        let (rank_sum, tie) = csr_rank_sum_stats_two_groups(
            &row_ptr,
            &col_indices,
            &data,
            a.len(),
            a.len() + b.len(),
            1,
        )[0];
        let n1 = a.len() as f64;
        (rank_sum - n1 * (n1 + 1.0) / 2.0, tie)
    }

    /// Run [GeneGroupStats] on one gene column.
    fn group_stats(column: &[f32], lookup: &[u16], sizes: &[usize]) -> GeneGroupStats {
        let mut indices: Vec<u32> = Vec::new();
        let mut data: Vec<F16> = Vec::new();
        for (cell, &v) in column.iter().enumerate() {
            if v != 0.0 {
                indices.push(cell as u32);
                data.push(F16::from_f32(v));
            }
        }

        let mut stats = GeneGroupStats::new(sizes.len());
        stats.gather(&data, &indices, lookup);
        stats.sweep(sizes);
        stats
    }

    /// Every pair and every one-vs-rest out of the single sweep must match the
    /// two-group kernel run on those groups alone, U and tie term both, with
    /// ties inside and across groups, zeros, and an ungrouped cell.
    #[test]
    fn test_group_stats_matches_two_group_kernel() {
        // 10 cells, cell 9 in no group
        let groups: Vec<Vec<usize>> = vec![vec![0, 1, 2], vec![3, 4, 5, 6], vec![7, 8]];
        let mut lookup = vec![NO_GROUP; 10];
        for (g, cells) in groups.iter().enumerate() {
            for &c in cells {
                lookup[c] = g as u16;
            }
        }
        let sizes: Vec<usize> = groups.iter().map(|g| g.len()).collect();

        let columns: [[f32; 10]; 4] = [
            [1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 3.0, 2.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            [0.0, 7.0, 6.0, 1.0, 2.0, 3.0, 0.0, 0.0, 9.0, 0.0],
        ];

        for (gene, column) in columns.iter().enumerate() {
            let stats = group_stats(column, &lookup, &sizes);

            for g in 0..3 {
                for h in (0..3).filter(|&h| h != g) {
                    let (u, tie) = two_group_reference(column, &groups[g], &groups[h]);
                    assert_relative_eq!(stats.u_pair(g, h), u, epsilon = 1e-12);
                    assert_relative_eq!(stats.tie_pair(g, h), tie, epsilon = 1e-12);
                }

                let rest: Vec<usize> = (0..3)
                    .filter(|&h| h != g)
                    .flat_map(|h| groups[h].iter().copied())
                    .collect();
                let (u, tie) = two_group_reference(column, &groups[g], &rest);
                assert_relative_eq!(stats.u_rest(g), u, epsilon = 1e-12);
                assert_relative_eq!(stats.tie_pooled(), tie, epsilon = 1e-12);
            }

            // the ungrouped cell 9 never reaches the counts
            let nnz: u64 = stats.nnz.iter().sum();
            let expected = column[..9].iter().filter(|&&v| v != 0.0).count() as u64;
            assert_eq!(nnz, expected, "gene {gene}");
        }
    }

    /// Reusing one instance across genes must not leak state from the
    /// previous gene.
    #[test]
    fn test_group_stats_scratch_reuse() {
        let lookup = vec![0_u16, 0, 1, 1];
        let sizes = [2_usize, 2];
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [4.0_f32, 3.0, 0.0, 1.0];

        let fresh = group_stats(&b, &lookup, &sizes);

        let mut reused = GeneGroupStats::new(2);
        for column in [&a, &b] {
            let mut indices: Vec<u32> = Vec::new();
            let mut data: Vec<F16> = Vec::new();
            for (cell, &v) in column.iter().enumerate() {
                if v != 0.0 {
                    indices.push(cell as u32);
                    data.push(F16::from_f32(v));
                }
            }
            reused.gather(&data, &indices, &lookup);
            reused.sweep(&sizes);
        }

        assert_eq!(reused.u_pair(0, 1), fresh.u_pair(0, 1));
        assert_eq!(reused.tie_pair(0, 1), fresh.tie_pair(0, 1));
        assert_eq!(reused.tie_pooled(), fresh.tie_pooled());
        assert_eq!(reused.nnz, fresh.nnz);
        assert_eq!(reused.sum, fresh.sum);
        // group 0 sits entirely above group 1
        assert_eq!(fresh.u_pair(0, 1), 4.0);
    }

    /// Appending is incremental and truncating back to a recorded prefix must
    /// restore the earlier state exactly, which is what lets a batch be undone.
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
