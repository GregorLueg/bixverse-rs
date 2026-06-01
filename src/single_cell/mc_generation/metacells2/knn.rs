//! Balanced KNN graph construction from a cell-cell similarity matrix.
//!
//! ### Algorithm
//!
//! 1. **Outgoing ranks**: for each row of the similarity matrix, rank entries
//!    by descending similarity (rank 1 = most similar). Diagonal is forced
//!    to the worst rank (`n`) to exclude self-loops.
//! 2. **Geometric-mean balancing**: `balanced[i,j] = sqrt(rank[i,j] * rank[j,i])`.
//!    Symmetric by construction. Lower = stronger edge.
//! 3. **Threshold and store**: drop entries with `balanced > max_rank` where
//!    `max_rank = k * balanced_ranks_factor`. Stored value for the surviving
//!    entries is `max_rank + 1 - balanced`. So balanced rank 1 stores as
//!    `max_rank` (largest weight); balanced rank `max_rank` stores as `1`
//!    (smallest non-zero weight). This inversion is what makes the final
//!    L1 row normalisation weight stronger edges more heavily.
//! 4. **Preserve best edge per row**: the highest-stored entry per row (by
//!    construction the entry with smallest balanced rank, excluding the
//!    diagonal) is recorded with a floor of 1.0 and re-added after the
//!    prunes. This guarantees min outgoing degree = 1 even if a cell has
//!    only weak neighbours.
//! 5. **Prune per column** (incoming): keep the largest `k * incoming_factor`
//!    stored values per column.
//! 6. **Prune per row** (outgoing): keep the largest `k * outgoing_factor`
//!    stored values per row.
//! 7. **Re-merge preserved**: element-wise max with the preserved entries
//!    and their transposes, ensuring bidirectional preservation.
//! 8. **L1 row normalise**: each row of the final matrix sums to 1.

use std::cmp::Ordering;

use faer::Mat;
use rayon::prelude::*;

use crate::core::math::sparse::coo_to_csr_presorted;
use crate::prelude::*;

use super::params::MC2KnnParams;

///////////
// Types //
///////////

/// Per-row scan output: column indices, stored values, and (best_col, best_val).
type RowScan = (Vec<usize>, Vec<f32>, (usize, f32));

/// Best-edge triple per row: (row, col, stored_value).
type PreservedEdge = (usize, usize, f32);

/////////////
// Helpers //
/////////////

/// Rank each row of `similarity` by descending value.
///
/// The diagonal is forced to `n` (worst rank) to exclude self-loops. Ties
/// are broken arbitrarily by the unstable sort.
///
/// ### Params
///
/// * `similarity` - Square `n × n` similarity matrix.
///
/// ### Returns
///
/// A dense `n × n` matrix of `u32` ranks where entry `(i, j)` is the rank
/// of column `j` within row `i` (rank 1 = most similar).
fn rank_rows_descending(similarity: &Mat<f32>) -> Mat<u32> {
    let n = similarity.nrows();
    let mut ranks = Mat::<u32>::from_fn(n, n, |_, _| 0);

    let col_stride = ranks.col_stride() as usize;
    let ranks_addr = ranks.as_ptr_mut() as usize;

    (0..n).into_par_iter().for_each(|i| {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| {
            similarity[(i, b)]
                .partial_cmp(&similarity[(i, a)])
                .unwrap_or(Ordering::Equal)
        });

        // SAFETY: each task owns a unique row `i`; writes at offset
        // `col * col_stride + i` are disjoint across tasks because every
        // task writes to a different `i`. The `Mat` outlives all tasks.
        unsafe {
            let ranks_ptr = ranks_addr as *mut u32;
            for (rank_minus_one, &col) in order.iter().enumerate() {
                *ranks_ptr.add(col * col_stride + i) = (rank_minus_one as u32) + 1;
            }
            *ranks_ptr.add(i * col_stride + i) = n as u32;
        }
    });

    ranks
}

/// Combine balancing, thresholding, and best-edge preservation in one pass.
///
/// Computes the geometric-mean balanced rank for each `(i, j)` pair, drops
/// entries exceeding `max_rank`, and converts survivors to stored values via
/// `max_rank + 1 - balanced`. Records each row's strongest non-self edge for
/// later re-insertion, floored at `1.0`.
///
/// ### Params
///
/// * `ranks` - Dense `n × n` rank matrix from `rank_rows_descending`.
/// * `max_rank` - Balanced-rank cutoff; entries at or above this are dropped.
///
/// ### Returns
///
/// A tuple of:
/// * CSR matrix of thresholded stored values, shape `(n, n)`.
/// * One `(row, col, stored_value)` triple per row recording the best
///   surviving edge, for use in `merge_max_with_preserved`.
fn balance_threshold_and_preserve(
    ranks: &Mat<u32>,
    max_rank: u32,
) -> (CompressedSparseData2<f32, f32>, Vec<PreservedEdge>) {
    let n = ranks.nrows();
    let max_rank_f = max_rank as f32;
    let cutoff = max_rank_f + 1.0;

    // Per-row scan: build sorted (col, stored) lists and find each row's
    // argmax. Returned rows are already sorted by column index.
    let per_row: Vec<RowScan> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut cols: Vec<usize> = Vec::new();
            let mut vals: Vec<f32> = Vec::new();
            let mut best_col = 0usize;
            let mut best_val = f32::NEG_INFINITY;

            for j in 0..n {
                if i == j {
                    continue;
                }
                let r_ij = ranks[(i, j)];
                let r_ji = ranks[(j, i)];
                let balanced = ((r_ij as f64) * (r_ji as f64)).sqrt() as f32;
                if balanced >= cutoff {
                    continue;
                }
                let stored = cutoff - balanced;
                if stored > best_val {
                    best_val = stored;
                    best_col = j;
                }
                cols.push(j);
                vals.push(stored);
            }

            // Even if no edge survived the threshold, every row gets a
            // preserved entry. We have to pick *some* column; scan for the
            // smallest balanced rank in the row (excluding self).
            if best_val == f32::NEG_INFINITY {
                let mut min_balanced = f32::INFINITY;
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let r_ij = ranks[(i, j)];
                    let r_ji = ranks[(j, i)];
                    let balanced = ((r_ij as f64) * (r_ji as f64)).sqrt() as f32;
                    if balanced < min_balanced {
                        min_balanced = balanced;
                        best_col = j;
                    }
                }
                best_val = 1.0; // floor for preservation
            } else {
                best_val = best_val.max(1.0);
            }

            (cols, vals, (best_col, best_val))
        })
        .collect();

    // assemble CSR directly (rows already sorted by column).
    let total_nnz: usize = per_row.iter().map(|(c, _, _)| c.len()).sum();
    let mut data = Vec::with_capacity(total_nnz);
    let mut indices = Vec::with_capacity(total_nnz);
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0);

    let mut preserved = Vec::with_capacity(n);
    for (i, (cols, vals, (bc, bv))) in per_row.into_iter().enumerate() {
        for (c, v) in cols.into_iter().zip(vals) {
            indices.push(c);
            data.push(v);
        }
        indptr.push(data.len());
        preserved.push((i, bc, bv));
    }

    let sparse = CompressedSparseData2 {
        data,
        indices: indices.index_cast(),
        indptr: indptr.index_cast(),
        cs_type: CompressedSparseFormat::Csr,
        data_2: None,
        shape: (n, n),
    };
    (sparse, preserved)
}

/// Keep only the top-`degree` entries per row by stored value.
///
/// Rows with fewer than `degree` entries are kept whole. Column order is
/// preserved within each row in the output.
///
/// ### Params
///
/// * `mat` - CSR matrix to prune.
/// * `degree` - Maximum number of entries to retain per row.
///
/// ### Returns
///
/// A new CSR matrix of the same shape with at most `degree` entries per row.
fn prune_per_row(
    mat: &CompressedSparseData2<f32, f32>,
    degree: usize,
) -> CompressedSparseData2<f32, f32> {
    debug_assert!(mat.cs_type.is_csr());
    let n_rows = mat.shape.0;

    // collect surviving (col, val) pairs per row, sorted by column.
    let per_row: Vec<Vec<(u32, f32)>> = (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let start = mat.indptr[i] as usize;
            let end = mat.indptr[i + 1] as usize;
            let nnz = end - start;
            if nnz <= degree {
                return mat.indices[start..end]
                    .iter()
                    .copied()
                    .zip(mat.data[start..end].iter().copied())
                    .collect();
            }

            let mut buf: Vec<(u32, f32)> = mat.indices[start..end]
                .iter()
                .copied()
                .zip(mat.data[start..end].iter().copied())
                .collect();

            // Partition so the first `degree` entries are the largest by value.
            // `select_nth_unstable_by` puts the nth largest at index
            // `degree - 1` if we order descending; everything before is
            // larger or equal.
            buf.select_nth_unstable_by(degree - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
            });
            buf.truncate(degree);
            buf.sort_unstable_by_key(|&(c, _)| c);
            buf
        })
        .collect();

    let total_nnz: usize = per_row.iter().map(|r| r.len()).sum();
    let mut data = Vec::with_capacity(total_nnz);
    let mut indices = Vec::with_capacity(total_nnz);
    let mut indptr = Vec::with_capacity(n_rows + 1);
    indptr.push(0);
    for row in per_row {
        for (c, v) in row {
            indices.push(c);
            data.push(v);
        }
        indptr.push(data.len() as u32);
    }

    CompressedSparseData2 {
        data,
        indices,
        indptr,
        cs_type: CompressedSparseFormat::Csr,
        data_2: None,
        shape: mat.shape,
    }
}

/// Keep only the top-`degree` entries per column by stored value.
///
/// Implemented as two transposes around `prune_per_row`; the matrix is never
/// materialised in CSC form beyond the intermediate buffers.
///
/// ### Params
///
/// * `mat` - CSR matrix to prune.
/// * `degree` - Maximum number of entries to retain per column.
///
/// ### Returns
///
/// A new CSR matrix of the same shape with at most `degree` entries per column.
fn prune_per_column(
    mat: &CompressedSparseData2<f32, f32>,
    degree: usize,
) -> CompressedSparseData2<f32, f32> {
    debug_assert!(mat.cs_type.is_csr());
    let transposed = mat.transpose_and_convert();
    let pruned = prune_per_row(&transposed, degree);
    pruned.transpose_and_convert()
}

/// Element-wise max of `pruned` with `preserved` entries and their transposes.
///
/// Each triple `(i, j, v)` contributes both `(i, j, v)` and `(j, i, v)`,
/// ensuring every cell retains at least one outgoing edge and appears as at
/// least one other cell's neighbour after pruning. Duplicate coordinates are
/// resolved by taking the larger value.
///
/// ### Params
///
/// * `pruned` - CSR matrix after both prune passes.
/// * `preserved` - Per-row best-edge triples from
///   `balance_threshold_and_preserve`.
/// * `n` - Matrix dimension.
///
/// ### Returns
///
/// A new CSR matrix of shape `(n, n)` with preserved edges merged in.
fn merge_max_with_preserved(
    pruned: CompressedSparseData2<f32, f32>,
    preserved: &[(usize, usize, f32)],
    n: usize,
) -> CompressedSparseData2<f32, f32> {
    let mut entries: Vec<(usize, usize, f32)> =
        Vec::with_capacity(pruned.indices.len() + preserved.len() * 2);

    for i in 0..n {
        let start = pruned.indptr[i] as usize;
        let end = pruned.indptr[i + 1] as usize;
        for idx in start..end {
            entries.push((i, pruned.indices[idx] as usize, pruned.data[idx]));
        }
    }
    for &(i, j, v) in preserved {
        entries.push((i, j, v));
        entries.push((j, i, v));
    }

    entries.sort_unstable_by_key(|a| (a.0, a.1));

    // dedup keeping max value within each (row, col) group.
    let mut deduped: Vec<(usize, usize, f32)> = Vec::with_capacity(entries.len());
    for entry in entries {
        match deduped.last_mut() {
            Some(last) if last.0 == entry.0 && last.1 == entry.1 => {
                if entry.2 > last.2 {
                    last.2 = entry.2;
                }
            }
            _ => deduped.push(entry),
        }
    }

    let rows: Vec<usize> = deduped.iter().map(|e| e.0).collect();
    let cols: Vec<usize> = deduped.iter().map(|e| e.1).collect();
    let vals: Vec<f32> = deduped.iter().map(|e| e.2).collect();

    coo_to_csr_presorted(&rows.index_cast(), &cols.index_cast(), &vals, (n, n))
}

/// L1-normalise each row in place so that each row sums to 1.
///
/// Rows with a zero sum are left unchanged. The normalisation makes stronger
/// edges (higher stored value) contribute more heavily downstream.
///
/// ### Params
///
/// * `mat` - CSR matrix to normalise; mutated in place.
///
/// ### Returns
///
/// The same matrix with each non-zero row divided by its row sum.
fn weigh_rows_l1(mut mat: CompressedSparseData2<f32, f32>) -> CompressedSparseData2<f32, f32> {
    debug_assert!(mat.cs_type.is_csr());
    let n_rows = mat.shape.0;
    for i in 0..n_rows {
        let start = mat.indptr[i] as usize;
        let end = mat.indptr[i + 1] as usize;
        let sum: f32 = mat.data[start..end].iter().sum();
        if sum > 0.0 {
            for v in &mut mat.data[start..end] {
                *v /= sum;
            }
        }
    }
    mat
}

//////////
// Main //
//////////

/// Build a weighted, asymmetric KNN graph from a square cell-cell similarity
/// matrix.
///
/// ### Params
///
/// * `similarity` - `n × n` symmetric similarity matrix. Entry `[i, i]` is
///   ignored (overwritten with worst rank during the rank pass).
/// * `k` - Target number of nearest neighbours per cell. Drives the prune
///   thresholds via `KnnParams::*_degree_factor` multipliers.
/// * `params` - KNN parameters; see [`KnnParams`].
///
/// ### Returns
///
/// CSR matrix of shape `(n, n)` with row-L1-normalised positive weights. No
/// self-loops.
///
/// ### Panics
///
/// Panics if `similarity` is non-square or if `k == 0`.
pub fn build_knn_graph(
    similarity: &Mat<f32>,
    k: usize,
    params: &MC2KnnParams,
) -> CompressedSparseData2<f32, f32> {
    let n = similarity.nrows();
    assert_eq!(similarity.ncols(), n, "similarity must be square");
    assert!(k > 0, "k must be positive");

    let max_rank = (k as f32 * params.balanced_ranks_factor).round().max(1.0) as u32;
    let incoming_degree =
        ((k as f32 * params.incoming_degree_factor).round() as usize).min(n.saturating_sub(1));
    let outgoing_degree = ((k as f32 * params.outgoing_degree_factor).round() as usize)
        .max(params.min_outgoing_degree)
        .min(n.saturating_sub(1));

    let ranks = rank_rows_descending(similarity);
    let (sparse, preserved) = balance_threshold_and_preserve(&ranks, max_rank);
    drop(ranks); // free n*n u32 before the sparse phase

    let pruned_in = prune_per_column(&sparse, incoming_degree);
    drop(sparse);

    let pruned_out = prune_per_row(&pruned_in, outgoing_degree);
    drop(pruned_in);

    let merged = merge_max_with_preserved(pruned_out, &preserved, n);
    weigh_rows_l1(merged)
}
