//! Highly variable gene selection and principal component analysis (PCA) for
//! meta cells. Works directly on `CompressedSparseData2` structures.

use faer::Mat;
use std::borrow::Cow;

use crate::core::base::info::parse_bin_strategy_type;
use crate::core::base::loess::LoessRegression;
use crate::core::math::pca_svd::*;
use crate::prelude::*;
use crate::single_cell::sc_processing::hvg::*;
use crate::single_cell::sc_processing::pca::{SingleCellPcaParams, SingleCellPcaRes};

/////////
// HVG //
/////////

/// HVG selection via VST from an in-memory sparse matrix.
///
/// Expects raw counts in `data` (no second layer needed). Accepts CSR or CSC
/// input -> if CSR, it transposes internally to CSC for column-wise iteration.
///
/// Shape must be (cells, genes).
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

    let (n_cells, n_genes) = csc.shape;
    let n_cells_f32 = n_cells as f32;
    let clip_max = clip_max.unwrap_or(n_cells_f32.sqrt());

    // pass 1: mean and variance per gene
    let mut means = Vec::with_capacity(n_genes);
    let mut vars = Vec::with_capacity(n_genes);

    for j in 0..n_genes {
        let start = csc.indptr[j] as usize;
        let end = csc.indptr[j + 1] as usize;
        let nnz = end - start;
        let slice = &csc.data[start..end];

        let sum: f32 = slice.iter().sum();
        let mean = sum / n_cells_f32;

        let ss_nonzero: f32 = slice.iter().map(|&x| (x - mean) * (x - mean)).sum();
        let ss_zeros = (n_cells - nnz) as f32 * mean * mean;
        let var = (ss_nonzero + ss_zeros) / n_cells_f32;

        means.push(mean);
        vars.push(var);
    }

    // loess fit on log10 scale
    let means_log10: Vec<f32> = means.iter().map(|x| x.log10()).collect();
    let vars_log10: Vec<f32> = vars.iter().map(|x| x.log10()).collect();

    let loess = LoessRegression::new(loess_span, 2);
    let loess_res = loess.fit(&means_log10, &vars_log10);

    // pass 2: standardised variance
    let mut var_standardised = Vec::with_capacity(n_genes);

    for j in 0..n_genes {
        let start = csc.indptr[j] as usize;
        let end = csc.indptr[j + 1] as usize;
        let nnz = end - start;
        let slice = &csc.data[start..end];

        let mean = means[j];
        let expected_var = 10_f32.powf(loess_res.fitted_vals[j]);
        let expected_sd = expected_var.sqrt();

        let mut sum_std = 0_f32;
        let mut sum_sq_std = 0_f32;

        for &val in slice {
            let norm = ((val - mean) / expected_sd).clamp(-clip_max, clip_max);
            sum_std += norm;
            sum_sq_std += norm * norm;
        }

        // zero entries
        let n_zeros = n_cells - nnz;
        if n_zeros > 0 {
            let std_zero = ((-mean) / expected_sd).clamp(-clip_max, clip_max);
            sum_std += n_zeros as f32 * std_zero;
            sum_sq_std += n_zeros as f32 * std_zero * std_zero;
        }

        let std_mean = sum_std / n_cells_f32;
        var_standardised.push((sum_sq_std / n_cells_f32) - (std_mean * std_mean));
    }

    HvgRes {
        mean: means.r_float_convert(),
        var: vars.r_float_convert(),
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
    let (n_cells, n_genes) = csc.shape;
    let n = n_cells as f64;
    let binning = parse_bin_strategy_type(binning).unwrap_or_default();

    let mut means = Vec::with_capacity(n_genes);
    let mut dispersions = Vec::with_capacity(n_genes);

    for j in 0..n_genes {
        let start = csc.indptr[j] as usize;
        let end = csc.indptr[j + 1] as usize;
        let slice = &csc.data[start..end];

        // zero entries contribute expm1(0) = 0 so nonzeros suffice
        let mut sum = 0f64;
        let mut sum_sq = 0f64;
        for &val in slice {
            let v = (val as f64).exp_m1();
            sum += v;
            sum_sq += v * v;
        }

        let mean = sum / n;
        let var = if n > 1.0 {
            ((sum_sq - n * mean * mean) / (n - 1.0)).max(0.0)
        } else {
            0.0
        };

        let exp_mean = mean.ln_1p() as f32;
        let log_vmr = if mean > 0.0 && var > 0.0 {
            (var / mean).ln() as f32
        } else {
            f32::NAN
        };

        means.push(exp_mean);
        dispersions.push(log_vmr);
    }

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
            .iter()
            .map(|&v| (v.exp_m1() * sf_inv).ln_1p())
            .collect();
        Cow::Owned(transformed)
    } else {
        Cow::Borrowed(vals.as_slice())
    };

    let row_offsets = if params_pca.clr { clr_offsets } else { None };

    let mut scaled = Mat::<f64>::zeros(n_cells, n_genes);

    for j in 0..n_genes {
        let start = csc.indptr[j] as usize;
        let end = csc.indptr[j + 1] as usize;
        let mut col = vec![0.0_f64; n_cells];
        for idx in start..end {
            let i = csc.indices[idx] as usize;
            col[i] = vals_active[idx] as f64;
        }
        if let Some(off) = row_offsets {
            for i in 0..n_cells {
                col[i] -= off[i];
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

        for i in 0..n_cells {
            let mut v = col[i];
            if params_pca.mean_center {
                v -= mean;
            }
            if params_pca.normalise_variance {
                v = if std_dev < 1e-8 { 0.0 } else { v / std_dev };
            }
            scaled[(i, j)] = v;
        }
    }

    let (scores, loadings, s) = if params_pca.randomised {
        let res: RandomSvdResults<f64> =
            randomised_svd(scaled.as_ref(), no_pcs, seed, Some(100_usize), None)?;
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
