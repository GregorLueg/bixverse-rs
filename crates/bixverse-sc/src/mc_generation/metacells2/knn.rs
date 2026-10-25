//! Balanced KNN graph construction from a cell-cell similarity matrix.
//!
//! Mirrors `metacells.tools.knn_graph.compute_obs_obs_knn_graph`, including
//! its `f32` arithmetic so that stored weights match upstream exactly.
//!
//! ### Algorithm
//!
//! 1. **Outgoing ranks**: rank each row by descending similarity (rank 1 =
//!    most similar). The diagonal gets the worst rank `n`.
//! 2. **Balance**: `stored = (2^21 - sqrt(r_ij r_ji)) + (max_rank + 1 - 2^21)`
//!    in `f32`, with `max_rank = k * balanced_ranks_factor`. Entries `<= 0` are
//!    dropped. The row's strongest entry is forced in with a floor of `1`.
//! 3. **Preserve**: each row's first maximum after step 2 is recorded.
//! 4. **Prune per column** (incoming): keep the largest
//!    `round(k * incoming_degree_factor)` stored values per column.
//! 5. **Prune per row** (outgoing): keep the largest
//!    `round(k * outgoing_degree_factor)` stored values per row.
//! 6. **Re-merge preserved**: element-wise max with the preserved entries and
//!    their transposes.
//! 7. **Retry**: if any row has fewer than `min_outgoing_degree` entries, raise
//!    `k` by `max(1, k / 10)` and redo steps 2 to 6.
//! 8. **L1 row normalise**.

use faer::Mat;
use rayon::prelude::*;

use crate::prelude::*;

use super::params::MC2KnnParams;

////////////
// Consts //
////////////

/// Rows handled per task when walking the column-major `n x n` matrices.
///
/// Sixteen `f32` or `u32` values fill one 64-byte cache line, so a tile read
/// down a column is one line per column.
const ROW_TILE: usize = 16;

/// Offset upstream adds before subtracting balanced ranks. In `f32` near
/// `2^21` the spacing is `1/8` to `1/4`, which quantises the stored values;
/// reproducing it keeps the weights identical to upstream's.
const BALANCE_SHIFT: f32 = 2_097_152.0;

/////////////
// Helpers //
/////////////

/// Rank each row of `similarity` by descending value.
///
/// The diagonal is forced to `n` (worst rank) to exclude self-loops. Ties
/// are broken arbitrarily by the unstable sort, as upstream.
///
/// ### Params
///
/// * `similarity` - Square `n × n` similarity matrix.
///
/// ### Returns
///
/// A dense `n × n` matrix of `u32` ranks, stored **transposed**: entry
/// `(j, i)` is the rank of column `j` within row `i` (rank 1 = most similar),
/// so each row's ranks are written down one contiguous column.
fn rank_rows_descending(similarity: &Mat<f32>) -> Mat<u32> {
    let n = similarity.nrows();
    let mut ranks = Mat::<u32>::from_fn(n, n, |_, _| 0);

    let col_stride = ranks.col_stride() as usize;
    let ranks_addr = ranks.as_ptr_mut() as usize;

    (0..n.div_ceil(ROW_TILE)).into_par_iter().for_each_init(
        || (Vec::<f32>::new(), Vec::<usize>::new()),
        |(panel, order), tile| {
            let first = tile * ROW_TILE;
            let rows = ROW_TILE.min(n - first);

            // gather the tile's rows: one cache line per column read
            panel.clear();
            panel.resize(rows * n, 0.0);
            for j in 0..n {
                for r in 0..rows {
                    panel[r * n + j] = similarity[(first + r, j)];
                }
            }

            for r in 0..rows {
                let i = first + r;
                let row = &panel[r * n..(r + 1) * n];
                order.clear();
                order.extend(0..n);
                order.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]));

                // SAFETY: each task owns the columns `first..first + rows`, so
                // writes at `i * col_stride + col` are disjoint across tasks.
                // The `Mat` outlives all tasks.
                unsafe {
                    let ranks_ptr = ranks_addr as *mut u32;
                    let mut rank = 1u32;
                    for &col in order.iter() {
                        if col != i {
                            *ranks_ptr.add(i * col_stride + col) = rank;
                            rank += 1;
                        }
                    }
                    *ranks_ptr.add(i * col_stride + i) = n as u32;
                }
            }
        },
    );

    ranks
}

/// Upstream's stored value for one pair: `(2^21 - sqrt(r_ij r_ji)) + offset`,
/// all in `f32`, with `offset = max_rank + 1 - 2^21` rounded once to `f32`.
///
/// ### Params
///
/// * `r_ij`, `r_ji` - Outgoing ranks in both directions.
/// * `offset` - `max_rank + 1 - 2^21` as `f32`.
///
/// ### Returns
///
/// The stored value; the edge survives when it is `> 0`.
#[inline]
fn balanced_stored(r_ij: u32, r_ji: u32, offset: f32) -> f32 {
    let balanced = ((r_ij as f32) * (r_ji as f32)).sqrt();
    (BALANCE_SHIFT - balanced) + offset
}

/// Balance, threshold and preserve, mirroring upstream's `_balance_ranks`.
///
/// For each row, stores `balanced_stored` for every off-diagonal pair that
/// comes out positive, and forces the row's strongest pair (first maximum)
/// into the matrix with a floor of `1`, so every row keeps an edge.
///
/// ### Params
///
/// * `ranks` - Dense `n × n` transposed rank matrix from
///   [`rank_rows_descending`].
/// * `max_rank` - `k * balanced_ranks_factor`, not rounded.
///
/// ### Returns
///
/// CSR matrix of stored values, shape `(n, n)`, columns sorted.
fn balance_ranks(ranks: &Mat<u32>, max_rank: f64) -> CompressedSparseData2<f32, f32> {
    let n = ranks.nrows();
    let offset = (max_rank + 1.0 - BALANCE_SHIFT as f64) as f32;

    // Rows are taken in tiles so that the strided `ranks[(i, j)]` is gathered
    // once per tile into contiguous panels.
    let per_row: Vec<Vec<(u32, f32)>> = (0..n.div_ceil(ROW_TILE))
        .into_par_iter()
        .map_init(Vec::<u32>::new, |panel, tile| {
            let first = tile * ROW_TILE;
            let rows = ROW_TILE.min(n - first);

            panel.clear();
            panel.resize(rows * n, 0);
            for j in 0..n {
                for r in 0..rows {
                    panel[r * n + j] = ranks[(first + r, j)];
                }
            }

            let mut out = Vec::with_capacity(rows);
            for r in 0..rows {
                let i = first + r;
                // transposed storage: column i holds rank[i, j], the panel
                // holds rank[j, i]
                let ranks_ji = &panel[r * n..(r + 1) * n];
                let col_i = ranks.col(i);

                let mut row: Vec<(u32, f32)> = Vec::new();
                let mut best: Option<(usize, f32)> = None;
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let stored = balanced_stored(col_i[j], ranks_ji[j], offset);
                    if best.is_none_or(|(_, v)| stored > v) {
                        best = Some((j, stored));
                    }
                    if stored > 0.0 {
                        row.push((j as u32, stored));
                    }
                }

                if let Some((best_col, best_val)) = best {
                    let floored = best_val.max(1.0);
                    match row.binary_search_by_key(&(best_col as u32), |&(c, _)| c) {
                        Ok(pos) => row[pos].1 = floored,
                        Err(pos) => row.insert(pos, (best_col as u32, floored)),
                    }
                }
                out.push(row);
            }
            out
        })
        .flatten_iter()
        .collect();

    rows_to_csr(per_row, (n, n))
}

/// Assemble per-row `(col, value)` lists, each sorted by column, into CSR.
///
/// ### Params
///
/// * `rows` - One list per row.
/// * `shape` - Matrix shape.
///
/// ### Returns
///
/// The CSR matrix.
fn rows_to_csr(
    rows: Vec<Vec<(u32, f32)>>,
    shape: (usize, usize),
) -> CompressedSparseData2<f32, f32> {
    let total_nnz: usize = rows.iter().map(|r| r.len()).sum();
    let mut data = Vec::with_capacity(total_nnz);
    let mut indices = Vec::with_capacity(total_nnz);
    let mut indptr = Vec::with_capacity(rows.len() + 1);
    indptr.push(0u32);
    for row in rows {
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
        shape,
    }
}

/// First column holding the row maximum, per row, as upstream's sparse
/// `argmax(axis=1)`.
///
/// ### Params
///
/// * `mat` - CSR matrix.
///
/// ### Returns
///
/// `(row, col, value)` per non-empty row.
fn row_argmax(mat: &CompressedSparseData2<f32, f32>) -> Vec<(usize, usize, f32)> {
    (0..mat.shape.0)
        .filter_map(|i| {
            let start = mat.indptr[i] as usize;
            let end = mat.indptr[i + 1] as usize;
            (start..end)
                .reduce(|b, idx| if mat.data[idx] > mat.data[b] { idx } else { b })
                .map(|b| (i, mat.indices[b] as usize, mat.data[b]))
        })
        .collect()
}

/// Keep only the top-`degree` entries per row by stored value.
///
/// Rows with at most `degree` entries are kept whole. Column order is
/// preserved within each row in the output. Ties at the cut are broken
/// arbitrarily, as upstream's `nth_element`.
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

    let per_row: Vec<Vec<(u32, f32)>> = (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let start = mat.indptr[i] as usize;
            let end = mat.indptr[i + 1] as usize;
            let mut buf: Vec<(u32, f32)> = mat.indices[start..end]
                .iter()
                .copied()
                .zip(mat.data[start..end].iter().copied())
                .collect();
            if buf.len() <= degree {
                return buf;
            }
            if degree == 0 {
                return Vec::new();
            }
            // the first `degree` entries become the largest by value
            buf.select_nth_unstable_by(degree - 1, |a, b| b.1.total_cmp(&a.1));
            buf.truncate(degree);
            buf.sort_unstable_by_key(|&(c, _)| c);
            buf
        })
        .collect();

    rows_to_csr(per_row, mat.shape)
}

/// Keep only the top-`degree` entries per column by stored value.
///
/// Implemented as two transposes around [`prune_per_row`].
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
/// Each triple `(i, j, v)` contributes both `(i, j, v)` and `(j, i, v)`, as
/// upstream's two `maximum` calls.
///
/// ### Params
///
/// * `pruned` - CSR matrix after both prune passes.
/// * `preserved` - Per-row strongest edges from [`row_argmax`].
///
/// ### Returns
///
/// A new CSR matrix with the preserved edges merged in.
fn merge_max_with_preserved(
    pruned: &CompressedSparseData2<f32, f32>,
    preserved: &[(usize, usize, f32)],
) -> CompressedSparseData2<f32, f32> {
    let n = pruned.shape.0;
    let mut extra: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
    for &(i, j, v) in preserved {
        extra[i].push((j as u32, v));
        extra[j].push((i as u32, v));
    }

    let rows: Vec<Vec<(u32, f32)>> = extra
        .into_par_iter()
        .enumerate()
        .map(|(i, mut row)| {
            let start = pruned.indptr[i] as usize;
            let end = pruned.indptr[i + 1] as usize;
            row.extend(
                pruned.indices[start..end]
                    .iter()
                    .copied()
                    .zip(pruned.data[start..end].iter().copied()),
            );
            row.sort_unstable_by_key(|&(c, _)| c);
            row.dedup_by(|later, kept| {
                let same = later.0 == kept.0;
                if same {
                    kept.1 = kept.1.max(later.1);
                }
                same
            });
            row
        })
        .collect();

    rows_to_csr(rows, (n, n))
}

/// L1-normalise each row in place so that each row sums to 1.
///
/// Rows with a zero sum are left unchanged. As upstream, each row is scaled
/// by the reciprocal of its sum.
///
/// ### Params
///
/// * `mat` - CSR matrix to normalise.
///
/// ### Returns
///
/// The same matrix with each non-zero row scaled to sum to one.
fn weigh_rows_l1(mut mat: CompressedSparseData2<f32, f32>) -> CompressedSparseData2<f32, f32> {
    debug_assert!(mat.cs_type.is_csr());
    let n_rows = mat.shape.0;
    for i in 0..n_rows {
        let start = mat.indptr[i] as usize;
        let end = mat.indptr[i + 1] as usize;
        let sum: f32 = mat.data[start..end].iter().sum();
        if sum > 0.0 {
            let scale = sum.recip();
            for v in &mut mat.data[start..end] {
                *v *= scale;
            }
        }
    }
    mat
}

/// One round of upstream's balance, preserve and prune for a given `k`.
///
/// ### Params
///
/// * `ranks` - Transposed rank matrix from [`rank_rows_descending`].
/// * `k` - Current `k`.
/// * `params` - KNN parameters.
///
/// ### Returns
///
/// The pruned, unnormalised CSR graph.
fn balance_and_prune(
    ranks: &Mat<u32>,
    k: usize,
    params: &MC2KnnParams,
) -> CompressedSparseData2<f32, f32> {
    let n = ranks.nrows();
    let degree =
        |factor: f64| ((k as f64 * factor).round_ties_even() as usize).min(n.saturating_sub(1));

    let balanced = balance_ranks(ranks, k as f64 * params.balanced_ranks_factor);
    let preserved = row_argmax(&balanced);
    let pruned_in = prune_per_column(&balanced, degree(params.incoming_degree_factor));
    drop(balanced);
    let pruned_out = prune_per_row(&pruned_in, degree(params.outgoing_degree_factor));
    drop(pruned_in);
    merge_max_with_preserved(&pruned_out, &preserved)
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
///   ignored.
/// * `k` - Target number of nearest neighbours per cell. Raised by 10% per
///   round while any cell has fewer than `params.min_outgoing_degree`
///   outgoing edges.
/// * `params` - KNN parameters; see [`MC2KnnParams`].
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

    // no cell can have more than n - 1 neighbours, so cap the requirement or
    // the retry never ends on tiny piles
    let required = params.min_outgoing_degree.min(n.saturating_sub(1));
    let ranks = rank_rows_descending(similarity);
    let mut k = k;
    loop {
        let pruned = balance_and_prune(&ranks, k, params);
        let min_degree = (0..n)
            .map(|i| (pruned.indptr[i + 1] - pruned.indptr[i]) as usize)
            .min()
            .unwrap_or(0);
        if min_degree >= required {
            return weigh_rows_l1(pruned);
        }
        k += (k / 10).max(1);
    }
}
