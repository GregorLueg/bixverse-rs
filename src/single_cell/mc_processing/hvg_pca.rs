//! Highly variable gene selection and principal component analysis (PCA) for
//! meta cells. Works directly on `CompressedSparseData2` structures.

use faer::{Mat, MatRef};
use rayon::prelude::*;
use std::borrow::Cow;

use crate::core::base::info::parse_bin_strategy_type;
use crate::core::base::loess::LoessRegression;
use crate::core::math::pca_svd::*;
use crate::prelude::*;
use crate::single_cell::sc_processing::hvg::*;
use crate::single_cell::sc_processing::pca::{SingleCellPcaParams, SingleCellPcaRes};
use crate::utils::simd::{sum_squared_dev_widen_simd_f32, sum_widen_simd_f32};

/////////
// HVG //
/////////

/// First-pass statistics for one gene of an in-memory sparse matrix.
///
/// The in-memory twin of [`GeneStats`], which cannot be reused here because it
/// carries the extremes as raw `u32` counts.
#[derive(Clone, Copy, Default)]
struct DenseGeneStats {
    /// Mean over every cell, zeros included.
    mean: f32,
    /// Population variance over every cell, zeros included.
    var: f32,
    /// Largest stored value. Zero for a gene with no counts.
    max: f32,
    /// Smallest value, counting the implicit zeros. Zero unless the gene is
    /// expressed in every cell.
    min: f32,
}

/// HVG selection via VST from an in-memory sparse matrix.
///
/// Expects raw counts in `data` (no second layer needed). Accepts CSR or CSC
/// input -> if CSR, it transposes internally to CSC for column-wise iteration.
///
/// Shape must be (cells, genes).
///
/// Both passes run gene-parallel and accumulate in `f64`: a metacell column
/// sums the counts of dozens of cells, so an `f32` accumulator walks past
/// `2^24` and starts dropping entries. The second pass only scans a gene when
/// the clip can actually reach one of its values, exactly as
/// [`run_hvg_vst`] does; everywhere else the standardised variance is
/// `var / expected_var` in closed form, because unclipped standardised values
/// sum to zero by construction.
///
/// ### Params
///
/// * `matrix` - The count data of the metacell
/// * `span` - The span parameter for the Loess function
/// * `clip_max` - Optional clipping for the data
///
/// ### Returns
///
/// The [HvgRes]
pub fn get_hvg_vst_from_sparse(
    matrix: &CompressedSparseData2<f32>,
    loess_span: f32,
    clip_max: Option<f32>,
) -> HvgRes {
    // ensure CSC so columns = genes
    let csc = match matrix.cs_type {
        CompressedSparseFormat::Csc => Cow::Borrowed(matrix),
        CompressedSparseFormat::Csr => Cow::Owned(matrix.transform()),
    };

    let n_cells = csc.shape.0;
    let n_cells_f64 = n_cells as f64;
    let clip_max = clip_max.unwrap_or((n_cells as f32).sqrt());

    // pass 1: mean, variance and the two extremes per gene
    let stats: Vec<DenseGeneStats> = csc
        .indptr
        .par_windows(2)
        .map(|window| {
            let slice = &csc.data[window[0] as usize..window[1] as usize];
            let nnz = slice.len();
            if nnz == 0 {
                return DenseGeneStats::default();
            }

            let mean = sum_widen_simd_f32(slice) / n_cells_f64;
            let ss_nonzero = sum_squared_dev_widen_simd_f32(slice, mean);
            let ss_zeros = (n_cells - nnz) as f64 * mean * mean;

            let (min, max) = slice
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                    (lo.min(v), hi.max(v))
                });

            DenseGeneStats {
                mean: mean as f32,
                var: ((ss_nonzero + ss_zeros) / n_cells_f64) as f32,
                max,
                min: if nnz < n_cells { 0.0 } else { min },
            }
        })
        .collect();

    // loess fit on log10 scale
    let means_log10: Vec<f32> = stats.par_iter().map(|s| s.mean.log10()).collect();
    let vars_log10: Vec<f32> = stats.par_iter().map(|s| s.var.log10()).collect();

    let loess = LoessRegression::new(loess_span, 2);
    let loess_res = loess.fit(&means_log10, &vars_log10);

    // pass 2: standardised variance, closed form wherever the clip cannot reach
    let var_standardised: Vec<f32> = csc
        .indptr
        .par_windows(2)
        .zip(stats.par_iter())
        .zip(loess_res.fitted_vals.par_iter())
        .map(|((window, stats), &fitted)| {
            let expected_var = 10_f32.powf(fitted);

            if !clip_is_reachable(stats.mean, stats.min, stats.max, expected_var, clip_max) {
                return stats.var / expected_var;
            }

            let slice = &csc.data[window[0] as usize..window[1] as usize];
            let expected_sd = expected_var.sqrt();
            let mean = stats.mean;

            // `min`/`max` rather than `clamp`: the former discards NaN, the
            // latter propagates it, and a NaN in `var_std` panics the ranking
            // in `select_hvg`.
            let (mut sum_std, mut sum_sq_std) = slice.iter().fold((0f64, 0f64), |(s, sq), &val| {
                let norm = ((val - mean) / expected_sd).min(clip_max).max(-clip_max) as f64;
                (s + norm, sq + norm * norm)
            });

            // zero entries
            let n_zeros = n_cells - slice.len();
            if n_zeros > 0 {
                let std_zero = ((-mean) / expected_sd).min(clip_max).max(-clip_max) as f64;
                sum_std += n_zeros as f64 * std_zero;
                sum_sq_std += n_zeros as f64 * std_zero * std_zero;
            }

            let std_mean = sum_std / n_cells_f64;
            ((sum_sq_std / n_cells_f64) - std_mean * std_mean) as f32
        })
        .collect();

    HvgRes {
        mean: stats.iter().map(|s| s.mean as f64).collect(),
        var: stats.iter().map(|s| s.var as f64).collect(),
        var_exp: loess_res.fitted_vals.r_float_convert(),
        var_std: var_standardised.r_float_convert(),
    }
}

/// Dispersion-based HVG selection from an in-memory sparse matrix.
///
/// Expects log-normalised data in `data` (i.e. `log1p(x/sum * size_factor)`).
/// Back-transforms via `expm1` internally to match Seurat's
/// `FastExpMean`/`FastLogVMR`.
///
/// ### Params
///
/// * `matrix` - The count data of the metacell
/// * `binning` - The binning strategy to use. One of `"equal_width"` or
///   `"equal_freq"`.
/// * `n_bins` - Number of bins
///
/// ### Returns
///
/// The [HvgDispersionRes]
pub fn get_hvg_dispersion_from_sparse(
    matrix: &CompressedSparseData2<f32>,
    binning: &str,
    n_bins: usize,
) -> HvgDispersionRes {
    let csc = match matrix.cs_type {
        CompressedSparseFormat::Csc => Cow::Borrowed(matrix),
        CompressedSparseFormat::Csr => Cow::Owned(matrix.transform()),
    };
    let n_cells = csc.shape.0;
    let binning = parse_bin_strategy_type(binning).unwrap_or_default();

    let (means, dispersions): (Vec<f32>, Vec<f32>) = csc
        .indptr
        .par_windows(2)
        .map(|window| {
            let slice = &csc.data[window[0] as usize..window[1] as usize];

            // zero entries contribute expm1(0) = 0 so nonzeros suffice
            let (sum, sum_sq) = slice.iter().fold((0f64, 0f64), |(s, sq), &val| {
                let v = (val as f64).exp_m1();
                (s + v, sq + v * v)
            });

            disp_stats_from_sums(sum, sum_sq, n_cells)
        })
        .unzip();

    build_disp_result(means, dispersions, binning, n_bins)
}

/// MVB HVG selection from an in-memory sparse matrix.
///
/// Computationally identical to `get_hvg_dispersion_from_sparse`; selection
/// logic (mean/dispersion cutoffs vs. top-N by dispersion) lives on the R side.
///
/// ### Params
///
/// * `matrix` - The count data of the metacell
/// * `binning` - The binning strategy to use. One of `"equal_width"` or
///   `"equal_freq"`.
/// * `n_bins` - Number of bins
///
/// ### Returns
///
/// The [HvgDispersionRes]
pub fn get_hvg_mvb_from_sparse(
    matrix: &CompressedSparseData2<f32>,
    binning: &str,
    n_bins: usize,
) -> HvgDispersionRes {
    get_hvg_dispersion_from_sparse(matrix, binning, n_bins)
}

/////////
// PCA //
/////////

/// PCA on pre-selected HVGs from an in-memory sparse matrix.
///
/// Reads normalised counts from the `data_2` layer (raw counts in `data`
/// are ignored). Densifies, optionally applies the shifted CLR
/// transformation, scales according to `params_pca`, then runs SVD.
/// Shape must be (cells, genes). Uses f64 internally for numerical
/// stability during SVD.
///
/// When `params_pca.clr` is true, `clr_offsets` must be provided and must
/// have been computed against the full gene panel (not the HVG subset),
/// since the row-mean of `log1p(u_ij)` cannot be reconstructed from a
/// gene subset alone.
///
/// ### Params
///
/// * `matrix` - The sparse counts. The `data_2` layer must be populated
///   with `log1p(u * sf)` normalised values.
/// * `no_pcs` - Number of PCs to return.
/// * `params_pca` - PCA parameters, see [SingleCellPcaParams].
/// * `clr_offsets` - Per-cell CLR offsets, required if `params_pca.clr`
///   is true. Length must equal the number of cells.
/// * `seed` - Random seed for the randomised SVD.
///
/// ### Returns
///
/// Tuple of (PCA scores, PCA loadings, singular values).
pub fn pca_on_metacells<T: BixverseNumeric>(
    matrix: &CompressedSparseData2<T, f32>,
    no_pcs: usize,
    params_pca: &SingleCellPcaParams,
    clr_offsets: Option<&[f64]>,
    seed: usize,
) -> SingleCellPcaRes {
    let (n_cells, n_genes) = matrix.shape;

    if params_pca.clr
        && let Some(offs) = clr_offsets
        && offs.len() != n_cells
    {
        return Err(BixverseErrors::OffsetsLengthDoesNotMatchNCells {
            len_offset: offs.len(),
            n_cells,
        });
    }

    let csc = match matrix.cs_type {
        CompressedSparseFormat::Csc => Cow::Borrowed(matrix),
        CompressedSparseFormat::Csr => Cow::Owned(matrix.transform()),
    };
    let vals = csc
        .data_2
        .as_ref()
        .expect("pca_on_metacells requires normalised counts in data_2");

    let vals_active: Cow<[f32]> = if params_pca.clr {
        let sf_inv = 1.0_f32 / params_pca.size_factor;
        let transformed: Vec<f32> = vals
            .par_iter()
            .map(|&v| (v.exp_m1() * sf_inv).ln_1p())
            .collect();
        Cow::Owned(transformed)
    } else {
        Cow::Borrowed(vals.as_slice())
    };

    let row_offsets = if params_pca.clr { clr_offsets } else { None };

    // column major, so one chunk is one gene and the scatter needs no scratch
    let mut buffer = vec![0.0_f64; n_cells * n_genes];

    buffer
        .par_chunks_mut(n_cells)
        .zip(csc.indptr.par_windows(2))
        .for_each(|(col, window)| {
            for idx in window[0] as usize..window[1] as usize {
                col[csc.indices[idx] as usize] = vals_active[idx] as f64;
            }
            if let Some(off) = row_offsets {
                for (v, o) in col.iter_mut().zip(off.iter()) {
                    *v -= o;
                }
            }

            let need_mean = params_pca.mean_center || params_pca.normalise_variance;
            let mean = if need_mean {
                col.iter().sum::<f64>() / n_cells as f64
            } else {
                0.0
            };
            let std_dev = if params_pca.normalise_variance {
                let var: f64 = col
                    .iter()
                    .map(|&x| {
                        let d = x - mean;
                        d * d
                    })
                    .sum::<f64>()
                    / (n_cells as f64 - 1.0);
                var.max(0.0).sqrt()
            } else {
                1.0
            };

            for v in col.iter_mut() {
                if params_pca.mean_center {
                    *v -= mean;
                }
                if params_pca.normalise_variance {
                    *v = if std_dev < 1e-8 { 0.0 } else { *v / std_dev };
                }
            }
        });

    let scaled = MatRef::<f64>::from_column_major_slice(&buffer, n_cells, n_genes);

    let (scores, loadings, s) = if params_pca.randomised {
        let res: RandomSvdResults<f64> =
            randomised_svd(scaled, no_pcs, seed, Some(100_usize), None)?;
        let loadings = Mat::<f32>::from_fn(n_genes, no_pcs, |i, j| res.v[(i, j)] as f32);
        let scores = Mat::<f32>::from_fn(n_cells, no_pcs, |i, j| (res.u[(i, j)] * res.s[j]) as f32);
        let s: Vec<f32> = res.s[..no_pcs].iter().map(|&x| x as f32).collect();
        (scores, loadings, s)
    } else {
        let res = scaled
            .thin_svd()
            .map_err(|e| BixverseErrors::FaerSvdError(format!("{e:?}")))?;
        let loadings = Mat::<f32>::from_fn(n_genes, no_pcs, |i, j| res.V()[(i, j)] as f32);
        let scores = Mat::<f32>::from_fn(n_cells, no_pcs, |i, j| {
            (res.U()[(i, j)] * res.S().column_vector()[j]) as f32
        });
        let s: Vec<f32> = res
            .S()
            .column_vector()
            .iter()
            .take(no_pcs)
            .map(|&x| x as f32)
            .collect();
        (scores, loadings, s)
    };

    Ok((scores, loadings, s))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Six cells, four genes. Gene 1 and 3 are expressed everywhere, gene 2 in
    /// a single cell, so the implicit-zero handling and the `min` semantics
    /// both get exercised.
    fn toy_csc() -> CompressedSparseData2<f32> {
        let data: Vec<f32> = vec![
            3.0, 5.0, 2.0, // gene 0
            1.0, 2.0, 1.0, 4.0, 1.0, 3.0, // gene 1
            7.0, // gene 2
            10.0, 12.0, 9.0, 11.0, 10.0, 13.0, // gene 3
        ];
        let indices: Vec<u32> = vec![0, 2, 4, 0, 1, 2, 3, 4, 5, 5, 0, 1, 2, 3, 4, 5];
        let indptr: Vec<u32> = vec![0, 3, 9, 10, 16];

        CompressedSparseData2::new_csc(&data, &indices, &indptr, None::<&[f32]>, (6, 4))
    }

    /// Standardised variance the slow way: densify the gene, standardise every
    /// cell, clip, then take the population variance.
    fn dense_var_std(
        matrix: &CompressedSparseData2<f32>,
        gene: usize,
        res: &HvgRes,
        clip: f32,
    ) -> f64 {
        let n_cells = matrix.shape.0;
        let mut dense = vec![0f32; n_cells];
        for idx in matrix.indptr[gene] as usize..matrix.indptr[gene + 1] as usize {
            dense[matrix.indices[idx] as usize] = matrix.data[idx];
        }

        let mean = res.mean[gene] as f32;
        let expected_sd = (10f64.powf(res.var_exp[gene]) as f32).sqrt();

        let (sum, sum_sq) = dense.iter().fold((0f64, 0f64), |(s, sq), &v| {
            let norm = ((v - mean) / expected_sd).min(clip).max(-clip) as f64;
            (s + norm, sq + norm * norm)
        });

        let n = n_cells as f64;
        (sum_sq / n) - (sum / n) * (sum / n)
    }

    /// Both branches of the second pass have to land on the dense reference:
    /// the closed form when the clip cannot reach any value, and the scan when
    /// it clips every single one.
    #[test]
    fn test_vst_clip_branches_match_the_dense_reference() {
        let matrix = toy_csc();

        for clip in [1e6_f32, 0.25] {
            let res = get_hvg_vst_from_sparse(&matrix, 0.3, Some(clip));

            for gene in 0..matrix.shape.1 {
                assert_relative_eq!(
                    res.var_std[gene],
                    dense_var_std(&matrix, gene, &res, clip),
                    epsilon = 1e-4
                );
            }
        }
    }

    /// `ExpMean` and `LogVMR` against the Seurat definitions, worked out on the
    /// densified column.
    #[test]
    fn test_dispersion_matches_the_seurat_definition() {
        let matrix = toy_csc();
        let res = get_hvg_dispersion_from_sparse(&matrix, "equal_width", 2);

        let n_cells = matrix.shape.0;
        for gene in 0..matrix.shape.1 {
            let mut dense = vec![0f64; n_cells];
            for idx in matrix.indptr[gene] as usize..matrix.indptr[gene + 1] as usize {
                dense[matrix.indices[idx] as usize] = matrix.data[idx] as f64;
            }

            let values: Vec<f64> = dense.iter().map(|v| v.exp_m1()).collect();
            let n = n_cells as f64;
            let mean = values.iter().sum::<f64>() / n;
            let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);

            assert_relative_eq!(res.mean[gene], mean.ln_1p(), epsilon = 1e-4);
            assert_relative_eq!(res.dispersion[gene], (var / mean).ln(), epsilon = 1e-4);
        }
    }
}
