//! Highly variable gene selection and principal component analysis (PCA) for
//! meta cells. Works directly on `CompressedSparseData2` structures.

use faer::Mat;
use std::borrow::Cow;

use crate::core::base::info::parse_bin_strategy_type;
use crate::core::base::loess::LoessRegression;
use crate::core::math::pca_svd::*;
use crate::prelude::*;
use crate::single_cell::sc_processing::hvg::*;
use crate::single_cell::sc_processing::pca::SingleCellPcaRes;

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
        let start = csc.indptr[j];
        let end = csc.indptr[j + 1];
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
        let start = csc.indptr[j];
        let end = csc.indptr[j + 1];
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
        let start = csc.indptr[j];
        let end = csc.indptr[j + 1];
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
/// are ignored). Densifies, scales (zero-mean, unit-variance per gene),
/// then runs SVD. Shape must be (cells, genes).
///
/// Uses f64 internally for numerical stability during SVD.
///
/// ### Params
///
/// * `matrix` - The sparse counts. The `data_2` layer must be populated with
///   normalised expression values.
/// * `no_pcs` - Number of PCs to return
/// * `random_svd` - Shall randomised SVD be used
/// * `seed` - Random seed for the randomised SVD
///
/// ### Returns
///
/// Tuple of (PCA scores, PCA loadings, singular values)
pub fn pca_on_metacells<T: BixverseNumeric>(
    matrix: &CompressedSparseData2<T, f32>,
    no_pcs: usize,
    random_svd: bool,
    seed: usize,
) -> SingleCellPcaRes {
    let (n_cells, n_genes) = matrix.shape;
    let csc = match matrix.cs_type {
        CompressedSparseFormat::Csc => Cow::Borrowed(matrix),
        CompressedSparseFormat::Csr => Cow::Owned(matrix.transform()),
    };
    let vals = csc
        .data_2
        .as_ref()
        .expect("pca_on_metacells requires normalised counts in data_2");

    let mut scaled = Mat::<f64>::zeros(n_cells, n_genes);
    for j in 0..n_genes {
        let start = csc.indptr[j];
        let end = csc.indptr[j + 1];
        for idx in start..end {
            let i = csc.indices[idx];
            scaled[(i, j)] = vals[idx] as f64;
        }
        let sum: f64 = (start..end).map(|idx| vals[idx] as f64).sum();
        let mean = sum / n_cells as f64;
        let nnz = end - start;
        let ss_nonzero: f64 = (start..end)
            .map(|idx| {
                let d = vals[idx] as f64 - mean;
                d * d
            })
            .sum();
        let ss_zeros = (n_cells - nnz) as f64 * mean * mean;
        let std_dev = ((ss_nonzero + ss_zeros) / (n_cells as f64 - 1.0))
            .max(0.0)
            .sqrt();
        if std_dev < 1e-8 {
            for i in 0..n_cells {
                scaled[(i, j)] = 0.0;
            }
        } else {
            for i in 0..n_cells {
                scaled[(i, j)] = (scaled[(i, j)] - mean) / std_dev;
            }
        }
    }

    let (scores, loadings, s) = if random_svd {
        let res: RandomSvdResults<f64> =
            randomised_svd(scaled.as_ref(), no_pcs, seed, Some(100_usize), None)?;
        let loadings = Mat::<f32>::from_fn(n_genes, no_pcs, |i, j| res.v[(i, j)] as f32);
        let scores = Mat::<f32>::from_fn(n_cells, no_pcs, |i, j| (res.u[(i, j)] * res.s[j]) as f32);
        let s: Vec<f32> = res.s[..no_pcs].iter().map(|&x| x as f32).collect();
        (scores, loadings, s)
    } else {
        let res = scaled.thin_svd().unwrap();
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
