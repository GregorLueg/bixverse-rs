//! Feature gene selection on the per-pile downsampled data.
//!
//! Three filters are applied to each gene:
//!
//! 1. **High total**: column sum across cells `>= min_gene_total`.
//! 2. **High top-N** (`N = 3`): the gene's N-th highest cell value
//!    `>= min_gene_topN`. A gene with fewer than N non-zero cells fails
//!    automatically.
//! 3. **High relative variance**: `log2(var/mean)` minus the median of
//!    `log2(var/mean)` over a centred window of genes with nearest mean
//!    `>= min_gene_relative_variance`.
//!
//! Active filters are ANDed; the optional lateral gene mask is then subtracted.
//! If fewer than `min_genes` survive, the relative-variance threshold is
//! stepped down (1/8 per step) until the count is met. If even at the floor the
//! count is short, the variance filter is dropped entirely.

use faer::Mat;
use rayon::prelude::*;

use crate::core::math::sparse::transpose_sparse;
use crate::prelude::*;

use super::params::SelectParams;
use super::pile::Pile;

/// "Top-3" filter: gene's 3rd-highest cell value must clear the threshold.
const TOP_N: usize = 3;

/// Per-iteration relaxation step on the relative-variance threshold when
/// fewer than `min_genes` survive at the configured threshold. Matches the
/// upstream `1 / 8` step. Below `RELAX_FLOOR` we give up and drop the
/// variance filter entirely.
const RELAX_STEP: f32 = 1.0 / 8.0;
const RELAX_FLOOR: f32 = -10.0;

/// Run the selection pipeline. Reads `pile.downsampled`; populates
/// `pile.selected_gene_indices` and `pile.selected_dense`.
///
/// ### Panics
///
/// Panics if `pile.downsampled` is `None`. Caller must run `downsample_pile`
/// first.
pub fn select_features(pile: &mut Pile, params: &SelectParams) {
    let downsampled = pile
        .downsampled
        .as_ref()
        .expect("downsample_pile must be called before select_features");
    let n_genes = downsampled.shape.1;

    // --- per-gene statistics (computed once, reused across threshold attempts).
    let total_per_gene = sum_columns_csr(downsampled);
    let top_n_per_gene = top_n_per_column(downsampled, TOP_N);
    let (mean_per_gene, log_norm_var_per_gene) = mean_and_log_norm_var(downsampled);
    let rel_var_per_gene = relative_variance(
        &log_norm_var_per_gene,
        &mean_per_gene,
        params.relative_variance_window_size,
    );

    // --- per-filter masks (None = filter disabled).
    let high_total = params.min_gene_total.map(|t| {
        total_per_gene
            .iter()
            .map(|&v| v >= t)
            .collect::<Vec<bool>>()
    });
    let high_top_n = params.min_gene_top3.map(|t| {
        top_n_per_gene
            .iter()
            .map(|&v| v >= t)
            .collect::<Vec<bool>>()
    });

    // --- variance filter with relaxation loop.
    let lateral = params.lateral_gene_mask.as_deref();

    let try_threshold = |t: f32| -> (Vec<bool>, usize) {
        let var_mask: Vec<bool> = rel_var_per_gene.iter().map(|&v| v >= t).collect();
        let combined = combine_masks(
            n_genes,
            high_total.as_deref(),
            high_top_n.as_deref(),
            Some(&var_mask),
            lateral,
        );
        let count = combined.iter().filter(|&&b| b).count();
        (combined, count)
    };

    let (mut mask, mut count) = match params.min_gene_relative_variance {
        Some(t) => try_threshold(t),
        None => {
            let m = combine_masks(
                n_genes,
                high_total.as_deref(),
                high_top_n.as_deref(),
                None,
                lateral,
            );
            let c = m.iter().filter(|&&b| b).count();
            (m, c)
        }
    };

    if count < params.min_genes
        && let Some(t0) = params.min_gene_relative_variance
    {
        let mut t = t0;
        while count < params.min_genes && t > RELAX_FLOOR {
            t -= RELAX_STEP;
            let (m, c) = try_threshold(t);
            mask = m;
            count = c;
        }
    }

    // If still short, drop the variance filter entirely.
    if count < params.min_genes {
        mask = combine_masks(
            n_genes,
            high_total.as_deref(),
            high_top_n.as_deref(),
            None,
            lateral,
        );
        // We accept whatever count is now; orchestrator decides if zero is fatal.
    }

    let selected: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i) } else { None })
        .collect();

    let dense = extract_dense_columns(downsampled, &selected);

    pile.selected_gene_indices = Some(selected);
    pile.selected_dense = Some(dense);
}

/// Sum the values of each column of a CSR matrix.
///
/// Uses saturating addition; true overflow is implausible on downsampled data
/// but the cost is nil.
///
/// ### Params
///
/// * `mat` - CSR matrix of `u32` counts.
///
/// ### Returns
///
/// A `Vec<u32>` of length `n_cols` with the column-wise totals.
fn sum_columns_csr(mat: &CompressedSparseData2<u32, f32>) -> Vec<u32> {
    let n_cols = mat.shape.1;
    let mut sums = vec![0u32; n_cols];
    for (idx, &col) in mat.indices.iter().enumerate() {
        sums[col] = sums[col].saturating_add(mat.data[idx]);
    }
    sums
}

/// Per-column N-th largest value, counting implicit zeros for sparse entries.
///
/// Transposes to CSC once, then runs `select_nth_unstable` per column in
/// parallel. A column with fewer than `n` non-zeros returns `0`, as the
/// implicit zeros would fill the bottom of the ranking.
///
/// ### Params
///
/// * `mat` - CSR matrix of `u32` counts.
/// * `n` - Rank to query (e.g. `3` for the 3rd-highest value).
///
/// ### Returns
///
/// A `Vec<u32>` of length `n_cols` with the N-th largest value per column.
fn top_n_per_column(mat: &CompressedSparseData2<u32, f32>, n: usize) -> Vec<u32> {
    debug_assert!(mat.cs_type.is_csr());
    let csc = transpose_sparse(mat);
    let n_genes = mat.shape.1;

    let csc_data = &csc.data;
    let csc_indptr = &csc.indptr;

    (0..n_genes)
        .into_par_iter()
        .map(|col| {
            let start = csc_indptr[col];
            let end = csc_indptr[col + 1];
            let nnz = end - start;
            if nnz < n {
                return 0u32;
            }
            // N-th largest = (nnz - n)-th smallest among the non-zeros.
            let mut buf: Vec<u32> = csc_data[start..end].to_vec();
            let pos = nnz - n;
            let (_, mid, _) = buf.select_nth_unstable(pos);
            *mid
        })
        .collect()
}

/// Per-column population mean and `log2(var/mean)`.
///
/// Accumulates in `f64` to guard against cancellation in `E[X²] - E[X]²`.
/// Returns `0.0` for `log_norm_var` when `mean == 0` (matching upstream's
/// `zero_value = 1.0`). Returns `NEG_INFINITY` when variance is zero but mean
/// is not, so the gene definitively fails the relative-variance filter.
///
/// ### Params
///
/// * `mat` - CSR matrix of `u32` counts.
///
/// ### Returns
///
/// A tuple `(mean, log_norm_var)`, each a `Vec<f32>` of length `n_cols`.
fn mean_and_log_norm_var(mat: &CompressedSparseData2<u32, f32>) -> (Vec<f32>, Vec<f32>) {
    let n_rows = mat.shape.0 as f64;
    let n_cols = mat.shape.1;

    let mut sum = vec![0.0f64; n_cols];
    let mut sum_sq = vec![0.0f64; n_cols];
    for (idx, &col) in mat.indices.iter().enumerate() {
        let v = mat.data[idx] as f64;
        sum[col] += v;
        sum_sq[col] += v * v;
    }

    let mut mean = vec![0.0f32; n_cols];
    let mut log_norm_var = vec![0.0f32; n_cols];

    if n_rows == 0.0 {
        return (mean, log_norm_var);
    }

    for col in 0..n_cols {
        let m = sum[col] / n_rows;
        mean[col] = m as f32;
        if m == 0.0 {
            log_norm_var[col] = 0.0; // log2(zero_value=1.0)
            continue;
        }
        let var = (sum_sq[col] / n_rows - m * m).max(0.0);
        let norm_var = var / m;
        log_norm_var[col] = if norm_var > 0.0 {
            (norm_var as f32).log2()
        } else {
            f32::NEG_INFINITY
        };
    }

    (mean, log_norm_var)
}

/// Per-gene relative variance: `log_norm_var[g]` minus the median of the window
/// of genes with nearest mean, ordered by mean ascending.
///
/// Window size is forced odd; near the edges the window is clipped rather
/// than padded or reflected.
///
/// ### Params
///
/// * `log_norm_var` - `log2(var/mean)` per gene.
/// * `mean` - Population mean per gene; used to define window order.
/// * `window_size` - Number of neighbours to include; forced odd internally.
///
/// ### Returns
///
/// A `Vec<f32>` of length `n` with the relative variance score per gene.
fn relative_variance(log_norm_var: &[f32], mean: &[f32], window_size: usize) -> Vec<f32> {
    let n = log_norm_var.len();
    let mut out = vec![0.0f32; n];
    if n == 0 {
        return out;
    }

    let w = if window_size.is_multiple_of(2) {
        window_size + 1
    } else {
        window_size
    };
    let half = w / 2;

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        mean[a]
            .partial_cmp(&mean[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut buf: Vec<f32> = Vec::with_capacity(w);
    for sort_pos in 0..n {
        let lo = sort_pos.saturating_sub(half);
        let hi = (sort_pos + half + 1).min(n);

        buf.clear();
        for k in lo..hi {
            buf.push(log_norm_var[order[k]]);
        }
        let med = median_inplace(&mut buf);
        let original_idx = order[sort_pos];
        out[original_idx] = log_norm_var[original_idx] - med;
    }

    out
}

/// In-place median via `select_nth_unstable_by`.
///
/// Returns the average of the two middle values for even-length input, matching
/// numpy's default. `NEG_INFINITY` and `NaN` are treated as equaq to any
/// non-comparable value; sufficient here as such entries either fail the
/// downstream threshold or land in stable positions.
///
/// ### Params
///
/// * `values` - Mutable slice of `f32`; partially reordered in place.
///
/// ### Returns
///
/// The median value, or `0.0` if `values` is empty.
fn median_inplace(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    let n = values.len();

    if n % 2 == 1 {
        let mid = n / 2;
        values.select_nth_unstable_by(mid, cmp);
        values[mid]
    } else {
        let upper = n / 2;
        values.select_nth_unstable_by(upper, cmp);
        let upper_val = values[upper];
        // After selection, values[..upper] are all <= upper_val. The lower
        // middle is their maximum.
        let mut lower_val = f32::NEG_INFINITY;
        for &v in &values[..upper] {
            if v > lower_val {
                lower_val = v;
            }
        }
        (lower_val + upper_val) / 2.0
    }
}

/// AND a set of optional boolean masks, then subtract the lateral mask.
///
/// `None` arguments are treated as all-true. The lateral mask is subtracted
/// last: genes set in it are excluded regardless of the other filters.
///
/// ### Params
///
/// * `n` - Length of the output mask.
/// * `a, b, c` - Optional per-gene filter masks; each `None` is a no-op.
/// * `lateral` - Optional lateral gene mask; matched genes are excluded.
///
/// ### Returns
///
/// A `Vec<bool>` of length `n` with the combined selection mask.
fn combine_masks(
    n: usize,
    a: Option<&[bool]>,
    b: Option<&[bool]>,
    c: Option<&[bool]>,
    lateral: Option<&[bool]>,
) -> Vec<bool> {
    let mut out = vec![true; n];
    for m in [a, b, c].iter().flatten() {
        for (o, &v) in out.iter_mut().zip(m.iter()) {
            *o &= v;
        }
    }
    if let Some(l) = lateral {
        for (o, &v) in out.iter_mut().zip(l.iter()) {
            *o &= !v;
        }
    }
    out
}

/// Extract selected columns from a CSR matrix into a dense
/// `n_cells × n_selected` matrix.
///
/// Builds a reverse-lookup from gene index to position in `selected`, then
/// walks the CSR rows once to fill non-zero entries. Unselected or absent
/// entries remain zero.
///
/// ### Params
///
/// * `csr` - Source CSR matrix of `u32` counts.
/// * `selected` - Sorted or unsorted slice of column indices to extract.
///
/// ### Returns
///
/// A `Mat<f32>` of shape `(n_cells, selected.len())`.
fn extract_dense_columns(csr: &CompressedSparseData2<u32, f32>, selected: &[usize]) -> Mat<f32> {
    let n_rows = csr.shape.0;
    let n_sel = selected.len();

    let mut gene_to_pos = vec![None; csr.shape.1];
    for (pos, &g) in selected.iter().enumerate() {
        gene_to_pos[g] = Some(pos);
    }

    let mut dense = Mat::<f32>::zeros(n_rows, n_sel);
    for row in 0..n_rows {
        let start = csr.indptr[row];
        let end = csr.indptr[row + 1];
        for idx in start..end {
            if let Some(pos) = gene_to_pos[csr.indices[idx]] {
                dense[(row, pos)] = csr.data[idx] as f32;
            }
        }
    }

    dense
}
