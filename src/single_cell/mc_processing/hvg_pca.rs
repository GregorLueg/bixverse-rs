//! Highly variable gene selection and principal component analysis (PCA).

use faer::Mat;
use std::borrow::Cow;

use crate::core::base::loess::LoessRegression;
use crate::core::math::pca_svd::*;
use crate::prelude::*;
use crate::single_cell::sc_processing::hvg::*;

/// HVG selection via VST from an in-memory sparse matrix.
///
/// Expects raw counts in `data` (no second layer needed). Accepts CSR or CSC
/// input -- if CSR, it transposes internally to CSC for column-wise iteration.
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
/// The HVG result
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

/// PCA on pre-selected HVGs from an in-memory sparse matrix.
///
/// Expects normalised counts in `data`. Densifies, scales (zero-mean,
/// unit-variance per gene), then runs SVD. Shape must be (cells, genes).
///
/// Uses f64 internally for numerical stability during SVD.
///
/// ### Params
///
/// * `matrix` - The sparse counts. Needs to have the second data layer!
/// * `no_pcs` - Number of PCs to return
/// * `random_svd` - Shall randomised SVD be used
/// * `seed` - Random seed for the randomised SVD
///
/// ### Returns
///
/// Tuple of (PCA scores, PCA loadings, singular values)
pub fn pca_on_metacells(
    matrix: &CompressedSparseData2<f32>,
    no_pcs: usize,
    random_svd: bool,
    seed: usize,
) -> (Mat<f32>, Mat<f32>, Vec<f32>) {
    let (n_cells, n_genes) = matrix.shape;

    // ensure CSC for column-wise densification
    let csc = match matrix.cs_type {
        CompressedSparseFormat::Csc => Cow::Borrowed(matrix),
        CompressedSparseFormat::Csr => Cow::Owned(matrix.transform()),
    };

    // densify and scale into f64
    let mut scaled = Mat::<f64>::zeros(n_cells, n_genes);

    for j in 0..n_genes {
        let start = csc.indptr[j];
        let end = csc.indptr[j + 1];

        // scatter non-zeros
        for idx in start..end {
            let i = csc.indices[idx];
            scaled[(i, j)] = csc.data[idx] as f64;
        }

        // compute mean
        let sum: f64 = (start..end).map(|idx| csc.data[idx] as f64).sum();
        let mean = sum / n_cells as f64;

        // compute std (n-1 denominator)
        let nnz = end - start;
        let ss_nonzero: f64 = (start..end)
            .map(|idx| {
                let d = csc.data[idx] as f64 - mean;
                d * d
            })
            .sum();
        let ss_zeros = (n_cells - nnz) as f64 * mean * mean;
        let std_dev = ((ss_nonzero + ss_zeros) / (n_cells as f64 - 1.0))
            .max(0.0)
            .sqrt();

        // scale column in-place
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

    // SVD
    let (scores, loadings, s) = if random_svd {
        let res: RandomSvdResults<f64> =
            randomised_svd(scaled.as_ref(), no_pcs, seed, Some(100_usize), None);
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

    (scores, loadings, s)
}
