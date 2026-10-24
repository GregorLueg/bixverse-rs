//! Feature gene selection on the per-pile downsampled data.
//!
//! Three filters are applied to each gene, as in `metacells.pipeline.select`:
//!
//! 1. **High total**: column sum across cells `>= min_gene_total`.
//! 2. **High top-3**: upstream's `topN = 3` statistic, which is the value at
//!    ascending rank `n - 3 - 1`, i.e. the 4th highest cell value counting
//!    implicit zeros, `>= min_gene_top3`.
//! 3. **High relative variance**: `log2(var/mean)` minus the sliding median of
//!    `log2(var/mean)` over genes ordered by mean, `>= min_gene_relative_variance`.
//!
//! Active filters are ANDed and the lateral gene mask is subtracted. If fewer
//! than `min_genes` survive, the relative variance threshold is stepped down
//! by 1/8 until enough pass and then bisected four times, unless the other
//! filters alone leave `<= min_genes`, in which case the variance filter is
//! dropped. All statistics come from one CSC transpose.

use faer::Mat;
use rayon::prelude::*;

use crate::prelude::*;
use bixverse_core::core::math::sparse::transpose_sparse;

use super::params::SelectParams;
use super::pile::Pile;

/// Upstream's `topN` for the top-3 filter. The statistic is the value at
/// ascending rank `n_cells - TOP_N - 1`, so it reads the 4th highest value.
const TOP_N: usize = 3;

/// Step of the relative variance relaxation, as upstream.
const RELAX_STEP: f64 = 1.0 / 8.0;

/// Bisection steps between the last failing and first passing threshold, as
/// upstream.
const RELAX_BISECTIONS: usize = 4;

/// Safety floor for the relaxation. Upstream loops without one, which only
/// terminates because enough genes have a finite relative variance. Below
/// this the variance filter is dropped instead of looping.
const RELAX_FLOOR: f64 = -64.0;

/// Fewest genes a correlation over cells can be computed on.
const MIN_CORRELATION_GENES: usize = 2;

//////////
// Main //
//////////

/// Run the selection pipeline. Reads `pile.downsampled`; populates
/// `pile.selected_gene_indices` and `pile.selected_dense`.
///
/// ### Params
///
/// * `pile` - Pile with `downsampled` populated.
/// * `params` - Selection parameters.
///
/// ### Returns
///
/// `Ok(())` on success, populating the pile.
///
/// ### Errors
///
/// * `Mc2LengthMismatch` if the lateral mask does not cover every gene.
/// * `Mc2TooFewGenes` if the variance filter is disabled and fewer than
///   `min_genes` pass (as upstream), or if fewer than two genes pass at all.
///
/// ### Panics
///
/// Panics if `pile.downsampled` is `None`. Caller must run `downsample_pile`
/// first.
pub fn select_features(pile: &mut Pile, params: &SelectParams) -> Result<(), ScErrors> {
    let downsampled = pile
        .downsampled
        .as_ref()
        .expect("downsample_pile must be called before select_features");
    let n_genes = downsampled.shape.1;

    if let Some(lateral) = params.lateral_gene_mask.as_deref()
        && lateral.len() != n_genes
    {
        return Err(ScErrors::Mc2LengthMismatch {
            what: "lateral_gene_mask",
            expected: n_genes,
            got: lateral.len(),
        });
    }

    let csc = transpose_sparse(downsampled);
    let stats = gene_stats(&csc);
    // upstream orders by its f32 means
    let mut order: Vec<usize> = (0..n_genes).collect();
    order.sort_by(|&a, &b| (stats.mean[a] as f32).total_cmp(&(stats.mean[b] as f32)));
    let rel_var = relative_variance(
        &stats.log_norm_var,
        &order,
        params.relative_variance_window_size,
    );

    // mask of the filters other than relative variance, lateral genes removed
    let base: Vec<bool> = (0..n_genes)
        .map(|g| {
            params
                .min_gene_total
                .is_none_or(|t| stats.total[g] >= t as f64)
                && params.min_gene_top3.is_none_or(|t| stats.top3[g] >= t)
                && !params.lateral_gene_mask.as_deref().is_some_and(|l| l[g])
        })
        .collect();
    let base_count = base.iter().filter(|&&b| b).count();

    let with_var = |t: f64| -> (Vec<bool>, usize) {
        let mask: Vec<bool> = base
            .iter()
            .zip(&rel_var)
            .map(|(&b, &v)| b && v >= t)
            .collect();
        let count = mask.iter().filter(|&&b| b).count();
        (mask, count)
    };

    let mask = match params.min_gene_relative_variance {
        None => {
            if base_count < params.min_genes {
                return Err(ScErrors::Mc2TooFewGenes {
                    selected: base_count,
                    required: params.min_genes,
                });
            }
            base.clone()
        }
        Some(t0) => {
            let t0 = t0 as f64;
            let (mask, count) = with_var(t0);
            if count >= params.min_genes {
                mask
            } else if base_count <= params.min_genes {
                base.clone()
            } else {
                relax_threshold(t0, params.min_genes, &with_var).unwrap_or_else(|| base.clone())
            }
        }
    };

    let selected: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| b.then_some(i))
        .collect();

    if selected.len() < MIN_CORRELATION_GENES {
        return Err(ScErrors::Mc2TooFewGenes {
            selected: selected.len(),
            required: MIN_CORRELATION_GENES,
        });
    }

    pile.selected_dense = Some(extract_dense_columns(&csc, &selected));
    pile.selected_gene_indices = Some(selected);

    Ok(())
}

/////////////
// Helpers //
/////////////

/// Per-gene statistics of the downsampled data.
pub(super) struct GeneStats {
    /// Column sums.
    pub(super) total: Vec<f64>,
    /// Upstream's top-3 statistic (4th highest value, implicit zeros counted).
    pub(super) top3: Vec<u32>,
    /// Column means over all cells.
    pub(super) mean: Vec<f64>,
    /// `log2(var / mean)`, `0` (`log2` of upstream's `zero_value = 1`) for an
    /// all-zero gene and `-inf` for a constant non-zero gene.
    pub(super) log_norm_var: Vec<f64>,
}

/// Compute every per-gene statistic in one parallel pass over the CSC view.
///
/// The variance uses upstream's `(sum_sq - sum^2 / n) / n` clipped at zero,
/// in `f64`.
///
/// ### Params
///
/// * `csc` - Downsampled counts, gene-major.
///
/// ### Returns
///
/// The [`GeneStats`].
pub(super) fn gene_stats(csc: &CompressedSparseData2<u32, f32>) -> GeneStats {
    let n_cells = csc.shape.0;
    let n_genes = csc.shape.1;
    let n = n_cells as f64;

    let per_gene: Vec<(f64, u32, f64, f64)> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let values = &csc.data[csc.indptr[g] as usize..csc.indptr[g + 1] as usize];
            let (sum, sum_sq) = values.iter().fold((0.0f64, 0.0f64), |(s, s2), &v| {
                let v = v as f64;
                (s + v, s2 + v * v)
            });
            let top3 = top_n_value(values, n_cells, TOP_N);
            if n_cells == 0 || sum == 0.0 {
                return (sum, top3, 0.0, 0.0);
            }
            let mean = sum / n;
            let var = ((sum_sq - sum * sum / n) / n).max(0.0);
            (sum, top3, mean, (var / mean).log2())
        })
        .collect();

    let mut stats = GeneStats {
        total: Vec::with_capacity(n_genes),
        top3: Vec::with_capacity(n_genes),
        mean: Vec::with_capacity(n_genes),
        log_norm_var: Vec::with_capacity(n_genes),
    };
    for (t, t3, m, l) in per_gene {
        stats.total.push(t);
        stats.top3.push(t3);
        stats.mean.push(m);
        stats.log_norm_var.push(l);
    }
    stats
}

/// Value at ascending rank `n_cells - top_n - 1` of a sparse column, implicit
/// zeros included.
///
/// ### Params
///
/// * `values` - Stored values of the column (may include explicit zeros).
/// * `n_cells` - Column length.
/// * `top_n` - Upstream's `topN`.
///
/// ### Returns
///
/// The value, `0` when the rank falls among the implicit zeros.
fn top_n_value(values: &[u32], n_cells: usize, top_n: usize) -> u32 {
    let above = top_n + 1;
    if values.len() < above || n_cells < above {
        return 0;
    }
    let mut buf = values.to_vec();
    let pos = buf.len() - above;
    *buf.select_nth_unstable(pos).1
}

/// Step the relative variance threshold down until `min_genes` pass, then
/// bisect between the last failing and first passing threshold.
///
/// ### Params
///
/// * `t0` - Configured threshold, known to fail.
/// * `min_genes` - Genes required.
/// * `with_var` - Mask and count at a threshold.
///
/// ### Returns
///
/// The last passing mask, or `None` if nothing passes above [`RELAX_FLOOR`].
fn relax_threshold(
    t0: f64,
    min_genes: usize,
    with_var: &impl Fn(f64) -> (Vec<bool>, usize),
) -> Option<Vec<bool>> {
    let mut low = t0;
    let (mut valid, mut high) = loop {
        let failing = low;
        low -= RELAX_STEP;
        if low < RELAX_FLOOR {
            return None;
        }
        let (mask, count) = with_var(low);
        if count >= min_genes {
            break (mask, failing);
        }
    };
    for _ in 0..RELAX_BISECTIONS {
        let mid = (high + low) / 2.0;
        let (mask, count) = with_var(mid);
        if count >= min_genes {
            low = mid;
            valid = mask;
        } else {
            high = mid;
        }
    }
    Some(valid)
}

/// Per-gene relative variance: `log_norm_var[g]` minus the median of a sliding
/// window over genes ordered by mean.
///
/// Mirrors `sliding_window_function` in `metacells.utilities.computation`:
/// the window is forced odd, a window covering every gene uses the global
/// median, and the edges are padded with sorted positions
/// `[w - h, w)` in front and `[n - w, n - w + h)` behind, not clipped.
///
/// ### Params
///
/// * `log_norm_var` - `log2(var/mean)` per gene.
/// * `order` - Genes by ascending mean. Upstream's order among equal means
///   is whatever numpy's unstable argsort yields; callers here use a stable
///   sort, so genes tied on the mean can see slightly different windows.
/// * `window_size` - Window size; forced odd.
///
/// ### Returns
///
/// The relative variance per gene.
pub(super) fn relative_variance(
    log_norm_var: &[f64],
    order: &[usize],
    window_size: usize,
) -> Vec<f64> {
    let n = log_norm_var.len();
    if n == 0 {
        return Vec::new();
    }
    let w = window_size | 1;
    let h = (w - 1) / 2;

    if w >= n {
        let med = median_inplace(&mut log_norm_var.to_vec());
        return log_norm_var.iter().map(|&v| v - med).collect();
    }

    let extended: Vec<f64> = (w - h..w)
        .chain(0..n)
        .chain(n - w..n - w + h)
        .map(|pos| log_norm_var[order[pos]])
        .collect();

    let mut out = vec![0.0; n];
    let mut buf = Vec::with_capacity(w);
    for (sort_pos, &gene) in order.iter().enumerate() {
        buf.clear();
        buf.extend_from_slice(&extended[sort_pos..sort_pos + w]);
        out[gene] = log_norm_var[gene] - median_inplace(&mut buf);
    }
    out
}

/// Median of an odd-length slice via `select_nth_unstable_by`, mean of the
/// two middle values for even length. Partially reorders the slice.
///
/// ### Params
///
/// * `values` - Values; `NaN` sorts last under `total_cmp`.
///
/// ### Returns
///
/// The median, `0` for an empty slice.
fn median_inplace(values: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let upper = n / 2;
    let upper_val = *values.select_nth_unstable_by(upper, f64::total_cmp).1;
    if n % 2 == 1 {
        return upper_val;
    }
    let lower_val = values[..upper]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    (lower_val + upper_val) / 2.0
}

/// Extract selected genes into a dense `n_cells × n_selected` matrix.
///
/// ### Params
///
/// * `csc` - Gene-major counts.
/// * `selected` - Gene indices, each at most once.
///
/// ### Returns
///
/// A column-major `Mat<f32>` of shape `(n_cells, selected.len())`.
fn extract_dense_columns(csc: &CompressedSparseData2<u32, f32>, selected: &[usize]) -> Mat<f32> {
    let mut dense = Mat::<f32>::zeros(csc.shape.0, selected.len());
    dense
        .par_col_iter_mut()
        .zip(selected.par_iter())
        .for_each(|(mut col, &g)| {
            for idx in csc.indptr[g] as usize..csc.indptr[g + 1] as usize {
                col[csc.indices[idx] as usize] = csc.data[idx] as f32;
            }
        });
    dense
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_n_value_counts_implicit_zeros() {
        // ten cells, five non-zeros: 4th highest is 2
        assert_eq!(top_n_value(&[9, 1, 7, 2, 5], 10, 3), 2);
        // three non-zeros: the 4th highest is an implicit zero
        assert_eq!(top_n_value(&[9, 7, 5], 10, 3), 0);
        // dense column of four: the 4th highest is the minimum
        assert_eq!(top_n_value(&[4, 3, 2, 1], 4, 3), 1);
    }

    #[test]
    fn test_relative_variance_full_window_uses_global_median() {
        let lnv = [1.0, 2.0, 3.0, 4.0, 5.0];
        let rv = relative_variance(&lnv, &[4, 3, 2, 1, 0], 100);
        assert_eq!(rv, vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relative_variance_pads_edges_like_upstream() {
        // n = 7, w = 3, h = 1, identity order: front pad is sorted position 2,
        // back pad is position 4
        let lnv: Vec<f64> = vec![0.0, 10.0, 1.0, 9.0, 2.0, 8.0, 3.0];
        let order: Vec<usize> = (0..7).collect();
        let rv = relative_variance(&lnv, &order, 3);
        // windows over [1, 0..7, 2]: [1,0,10] [0,10,1] [10,1,9] [1,9,2]
        // [9,2,8] [2,8,3] [8,3,2]
        let medians = [1.0, 1.0, 9.0, 2.0, 8.0, 3.0, 3.0];
        for g in 0..7 {
            assert_eq!(rv[g], lnv[g] - medians[g]);
        }
    }

    #[test]
    fn test_relax_threshold_bisects_to_last_passing() {
        // count passing at t = number of integers in [0, 10) with i >= 8 t
        let with_var = |t: f64| {
            let mask: Vec<bool> = (0..10).map(|i| i as f64 >= 8.0 * t).collect();
            let count = mask.iter().filter(|&&b| b).count();
            (mask, count)
        };
        // need 5 genes: pass when 8 t <= 5; first step from 1.0 down to 0.625
        // passes, bisection then pushes up between 0.625 and 0.75
        let mask = relax_threshold(1.0, 5, &with_var).unwrap();
        assert_eq!(mask.iter().filter(|&&b| b).count(), 5);
    }
}
