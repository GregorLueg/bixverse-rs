//! DSB normalisation for ADT (antibody-derived tag) counts.
//!
//! Implements both variants of the DSB algorithm (Mulè et al. 2022, Nat Comms):
//!
//! - With empty droplets: estimates per-protein ambient background from the
//!   empty drop counts and standardises (or mean-subtracts) cell counts against
//!   it.
//! - Without empty droplets ("ModelNegativeADTnorm"): fits a 2-component
//!   k-means per protein on the log-counts and uses the lower mean as the
//!   background level.
//!
//! Both variants share an optional Step II that removes cell-to-cell technical
//! noise by regressing out PC1 of a noise matrix built from isotype controls
//! (if available) and a per-cell background mean estimated by a 2-component
//! k-means on that cell's protein vector.

use faer::{Mat, MatRef};
use rayon::prelude::*;

use crate::ml::clustering::k_means::k_means_clusters;
use crate::prelude::*;

////////////////////
// Structs, Enums //
////////////////////

/// Step I -> Background scaling.
#[derive(Clone, Copy, Debug, Default)]
pub enum ScaleFactor {
    /// Subtract per-protein background mean, divide by per-protein background
    /// SD. Recommended default in the original implementation
    #[default]
    Standardise,
    /// Subtract per-protein background mean only.
    MeanSubtract,
}

/// Parse a scale factor.
///
/// ### Params
///
/// * `s` - String to parse
///
/// ### Returns
///
/// The option of the [ScaleFactor]
pub fn parse_scale_factor(s: &str) -> Option<ScaleFactor> {
    match s.to_lowercase().as_str() {
        "standardize" | "standardise" => Some(ScaleFactor::Standardise),
        "mean.subtract" | "mean_subtract" | "meansubtract" => Some(ScaleFactor::MeanSubtract),
        _ => None,
    }
}

/// Background estimation source for Step I.
pub enum BackgroundSource<'a> {
    /// Estimate background from empty droplets.
    EmptyDroplets {
        /// n_empty x n_proteins matrix of empty-drop ADT counts
        empty_counts: MatRef<'a, f32>,
        /// How to apply the per-protein background correction.
        scale_factor: ScaleFactor,
    },
    /// Estimate per-protein background from a 2-component clustering of
    /// cell-level log counts; use the lower mean. No SD available, so only mean
    /// subtraction is performed.
    ModelNegative,
}

/// Prepare the background source
///
/// ### Params
///
/// * `empty_counts` - An option of empty counts. If provided, it will use the
///   [BackgroundSource::EmptyDroplets] method.
/// * `scale_factor` - If empty_counts is Some this string will define the
///   background scaling
///
/// ### Returns
///
/// The [BackgroundSource]
pub fn prepare_background_source<'a>(
    empty_counts: Option<MatRef<'a, f32>>,
    scale_factor: String,
) -> BackgroundSource<'a> {
    if let Some(empty_counts) = empty_counts {
        let scale_factor = parse_scale_factor(&scale_factor).unwrap_or_default();

        BackgroundSource::EmptyDroplets {
            empty_counts,
            scale_factor,
        }
    } else {
        BackgroundSource::ModelNegative
    }
}

/// DSB result.
pub struct DsbResult {
    /// Normalised matrix, n_cells x n_proteins, f32.
    pub normalised: Mat<f32>,
    /// Per-protein background mean (length n_proteins).
    pub protein_background_mean: Vec<f64>,
    /// Per-protein background SD (length n_proteins). Only populated for
    /// the EmptyDroplets variant.
    pub protein_background_sd: Option<Vec<f64>>,
    /// Per-cell estimated technical component (length n_cells), if
    /// denoise_counts was true.
    pub technical_component: Option<Vec<f64>>,
    /// Per-cell background mean from the 2-component k-means clustering
    /// (length n_cells), if denoise_counts was true.
    pub cellwise_background_mean: Option<Vec<f64>>,
}

////////////
// Params //
////////////

/// DSB parameters.
#[derive(Clone, Debug)]
pub struct DsbParams {
    /// Run Step II (cell-to-cell technical noise removal).
    pub denoise_counts: bool,
    /// Include isotype controls in the noise matrix.
    pub use_isotype_controls: bool,
    /// Column indices of isotype control proteins (0-based, R-side resolution).
    pub isotype_indices: Vec<usize>,
    /// Pseudocount added before log transform. DSB recommends 10 for the
    /// EmptyDroplets variant and 1 for ModelNegative.
    pub pseudocount: f64,
    /// k-means iterations for Step II per-cell background estimation.
    pub kmeans_iters: usize,
    /// Optional quantile clipping per protein column: (low, high), both in
    /// [0, 1]. None disables clipping.
    pub quantile_clip: Option<(f64, f64)>,
}

/// Default implementation of [DsbParams]
impl Default for DsbParams {
    fn default() -> Self {
        Self {
            denoise_counts: true,
            use_isotype_controls: true,
            isotype_indices: Vec::new(),
            pseudocount: 10.0,
            kmeans_iters: 50,
            quantile_clip: None,
        }
    }
}

/////////////
// Helpers //
/////////////

/// Log-transform with pseudocount, returning an f64 working matrix.
///
/// Uses `f64` under the hood to avoid catastrophic cancellation
///
/// ### Params
///
/// * `mat` - The matrix to transform via `ln(x + pseudocount)`
/// * `pseudocount` - The pseudo count to use
///
/// ### Returns
///
/// The log transformed matrix. Outer vector -> rows
fn log_transform(mat: MatRef<f32>, pseudocount: f64) -> Vec<Vec<f64>> {
    let n_rows = mat.nrows();
    let n_cols = mat.ncols();
    (0..n_rows)
        .into_par_iter()
        .map(|i| {
            (0..n_cols)
                .map(|j| (mat[(i, j)] as f64 + pseudocount).ln())
                .collect()
        })
        .collect()
}

/// Per-column mean over rows.
///
/// ### Params
///
/// * `rows` - The matrix for which to calculate the column means
///
/// ### Returns
///
/// The column means
fn col_means(rows: &[Vec<f64>]) -> Vec<f64> {
    if rows.is_empty() {
        return Vec::new();
    }
    let n_cols = rows[0].len();
    let n_rows = rows.len() as f64;
    (0..n_cols)
        .into_par_iter()
        .map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / n_rows)
        .collect()
}

/// Per-column SD (sample SD, dividing by n - 1) over rows.
///
/// ### Params
///
/// * `rows` - The matrix for which to calculate the column means
/// * `means` - The column means
///
/// ### Returns
///
/// The column SDs
fn col_sds(rows: &[Vec<f64>], means: &[f64]) -> Vec<f64> {
    if rows.len() < 2 {
        return vec![0.0; means.len()];
    }
    let n_cols = means.len();
    let denom = (rows.len() - 1) as f64;
    (0..n_cols)
        .into_par_iter()
        .map(|j| {
            let s: f64 = rows.iter().map(|r| (r[j] - means[j]).powi(2)).sum();
            (s / denom).sqrt()
        })
        .collect()
}

/// Two-component k-means on a 1D column; return the lower centroid value.
///
/// ### Params
///
/// * `values` - The 1D column on which to apply the two-component k-means
/// * `n_iters` - Number of iterations to use for the k-means
/// * `seed` - The random seed for reproducibility
///
/// ### Returns
///
/// The lower k-means centroid
fn lower_centroid_kmeans(values: &[f64], n_iters: usize, seed: usize) -> f64 {
    let n = values.len();
    if n < 2 {
        return values.first().copied().unwrap_or(0.0);
    }
    // wrap as N x 1 f32 Mat for k_means_clusters
    let m = Mat::<f32>::from_fn(n, 1, |i, _| values[i] as f32);
    let (centroids, _assignments) =
        k_means_clusters(m.as_ref(), "euclidean", 2, n_iters, seed, false);
    let c0 = centroids[(0, 0)] as f64;
    let c1 = centroids[(1, 0)] as f64;
    c0.min(c1)
}

/// Per-protein quantile via sort and interp.
///
/// ### Params
///
/// * `rows` - The matrix
/// * `j` - The column index
/// * `q` - The quantile
///
/// ### Returns
///
/// The value at this quantile
fn col_quantile(rows: &[Vec<f64>], j: usize, q: f64) -> f64 {
    let mut vals: Vec<f64> = rows.iter().map(|r| r[j]).collect();
    let n = vals.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return vals[0];
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let h = (n as f64 - 1.0) * q;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = h - lo as f64;
    vals[lo] + frac * (vals[hi] - vals[lo])
}

//////////////////
// Step II core //
//////////////////

/// Compute the per-cell background mean (one 2-component k-means per cell on
/// its protein vector).
///
/// ### Params
///
/// * `cells` - Cells x proteins after the first step of the algorithm.
/// * `n_iters` - The number of iterations for the k-means clustering
/// * `seed` - The random seed for reproducibility
///
/// ### Returns
///
/// The cell-wise background means
fn cellwise_background_means(cells: &[Vec<f64>], n_iters: usize, seed: usize) -> Vec<f64> {
    cells
        .par_iter()
        .enumerate()
        .map(|(i, row)| lower_centroid_kmeans(row, n_iters, seed.wrapping_add(i)))
        .collect()
}

/// Build the noise matrix for Step II denoising.
///
/// Constructs a covariates x cells matrix from isotype control columns and
/// the per-cell background means. Each covariate row is z-scored (matching
/// `prcomp(scale = TRUE)` in R) before being returned.
///
/// ### Params
///
/// * `cells` - Cells x proteins matrix after Step I correction.
/// * `isotype_indices` - Column indices of isotype control proteins.
/// * `cellwise_bg` - Per-cell background means from k-means clustering.
///
/// ### Returns
///
/// A covariates x cells matrix of z-scored noise covariates.
fn build_noise_matrix(
    cells: &[Vec<f64>],
    isotype_indices: &[usize],
    cellwise_bg: &[f64],
) -> Vec<Vec<f64>> {
    let n_cells = cells.len();
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(isotype_indices.len() + 1);

    for &j in isotype_indices {
        let r: Vec<f64> = cells.iter().map(|c| c[j]).collect();
        rows.push(r);
    }
    rows.push(cellwise_bg.to_vec());

    // z-score each covariate (row)
    for row in rows.iter_mut() {
        let mean = row.iter().sum::<f64>() / n_cells as f64;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n_cells - 1) as f64;
        let sd = var.sqrt();
        if sd > 0.0 {
            for v in row.iter_mut() {
                *v = (*v - mean) / sd;
            }
        } else {
            for v in row.iter_mut() {
                *v -= mean;
            }
        }
    }
    rows
}

/// Compute PC1 scores of the noise matrix via thin SVD.
///
/// Takes a covariates x cells matrix (already z-scored) and returns the first
/// principal component scores per cell.
///
/// ### Params
///
/// * `noise_rows` - Covariates x cells matrix of z-scored noise covariates.
///
/// ### Returns
///
/// A length-n_cells vector of PC1 scores.
fn pc1_scores(noise_rows: &[Vec<f64>]) -> Result<Vec<f64>, BixverseErrors> {
    let n_cov = noise_rows.len();
    let n_cells = noise_rows[0].len();

    // build the matrix: rows = cells (samples), cols = covariates,
    let mat = Mat::<f64>::from_fn(n_cells, n_cov, |i, j| noise_rows[j][i]);

    let svd = mat.thin_svd().map_err(|_| BixverseErrors::FaerSvdError)?;
    let u = svd.U();
    let s = svd.S();

    // PC1 score for sample i = U[i, 0] * S[0]
    let s0 = s[0];
    let scores: Vec<f64> = (0..n_cells).map(|i| u[(i, 0)] * s0).collect();
    Ok(scores)
}

/// Regress a single covariate out of every protein column in-place.
///
/// For each protein j, estimates the OLS coefficient against `z` and subtracts
/// the fitted values, leaving only the residual signal.
///
/// ### Params
///
/// * `cells` - Cells x proteins matrix to modify in-place.
/// * `z` - The covariate to regress out (length n_cells).
fn regress_out_covariate(cells: &mut [Vec<f64>], z: &[f64]) {
    let n_cells = cells.len();
    if n_cells < 2 {
        return;
    }
    let n_cols = cells[0].len();
    let mean_z: f64 = z.iter().sum::<f64>() / n_cells as f64;
    let centred_z: Vec<f64> = z.iter().map(|v| v - mean_z).collect();
    let var_z: f64 = centred_z.iter().map(|v| v * v).sum::<f64>() / (n_cells - 1) as f64;

    if var_z <= 0.0 {
        return;
    }
    let denom = var_z * (n_cells - 1) as f64;

    // compute betas per column in parallel
    let betas: Vec<f64> = (0..n_cols)
        .into_par_iter()
        .map(|j| {
            let mean_y: f64 = cells.iter().map(|r| r[j]).sum::<f64>() / n_cells as f64;
            let num: f64 = cells
                .iter()
                .zip(centred_z.iter())
                .map(|(r, &zc)| (r[j] - mean_y) * zc)
                .sum();
            num / denom
        })
        .collect();

    // subtract beta_j * centred_z[i] from each cell, per protein
    cells.par_iter_mut().enumerate().for_each(|(i, row)| {
        let zc = centred_z[i];
        for j in 0..n_cols {
            row[j] -= betas[j] * zc;
        }
    });
}

//////////
// Main //
//////////

/// DSB normalise an ADT count matrix.
///
/// ### Params
///
/// * `cells` - n_cells x n_proteins matrix of raw ADT counts (f32).
/// * `background` - source for Step I background estimation.
/// * `params` - algorithm hyperparameters.
/// * `verbose` - print progress.
///
/// ### Returns
///
/// `DsbResult` containing the normalised matrix and per-protein/per-cell
/// diagnostics.
pub fn dsb_normalize(
    cells: MatRef<f32>,
    background: BackgroundSource,
    params: &DsbParams,
    verbose: bool,
    seed: usize,
) -> Result<DsbResult, BixverseErrors> {
    let n_cells = cells.nrows();
    let n_proteins = cells.ncols();

    if n_cells == 0 || n_proteins == 0 {
        return Err(BixverseErrors::InvalidArgument(
            "cells matrix is empty".to_string(),
        ));
    }
    for &j in &params.isotype_indices {
        if j >= n_proteins {
            return Err(BixverseErrors::InvalidArgument(format!(
                "isotype index {j} >= n_proteins {n_proteins}"
            )));
        }
    }

    // Step I: log + per-protein background correction
    let mut cell_log = log_transform(cells, params.pseudocount);

    let (bg_mean, bg_sd): (Vec<f64>, Option<Vec<f64>>) = match &background {
        BackgroundSource::EmptyDroplets {
            empty_counts,
            scale_factor,
        } => {
            if empty_counts.ncols() != n_proteins {
                return Err(BixverseErrors::InvalidArgument(format!(
                    "empty_counts has {} proteins, cells have {}",
                    empty_counts.ncols(),
                    n_proteins
                )));
            }
            let empty_log = log_transform(*empty_counts, params.pseudocount);
            let mu = col_means(&empty_log);
            let sd = col_sds(&empty_log, &mu);

            // apply: cell - mu_j (optionally / sd_j)
            match scale_factor {
                ScaleFactor::Standardise => {
                    cell_log.par_iter_mut().for_each(|row| {
                        for j in 0..n_proteins {
                            let denom = if sd[j] > 0.0 { sd[j] } else { 1.0 };
                            row[j] = (row[j] - mu[j]) / denom;
                        }
                    });
                    (mu, Some(sd))
                }
                ScaleFactor::MeanSubtract => {
                    cell_log.par_iter_mut().for_each(|row| {
                        for j in 0..n_proteins {
                            row[j] -= mu[j];
                        }
                    });
                    (mu, Some(sd))
                }
            }
        }
        BackgroundSource::ModelNegative => {
            if verbose {
                println!("DSB: estimating per-protein background via k-means (no empty drops)");
            }
            // 2-component k-means per protein column, take lower mean
            let mu: Vec<f64> = (0..n_proteins)
                .into_par_iter()
                .map(|j| {
                    let col: Vec<f64> = cell_log.iter().map(|r| r[j]).collect();
                    lower_centroid_kmeans(&col, params.kmeans_iters, seed.wrapping_add(j))
                })
                .collect();

            cell_log.par_iter_mut().for_each(|row| {
                for j in 0..n_proteins {
                    row[j] -= mu[j];
                }
            });
            (mu, None)
        }
    };

    // Step II: cell-to-cell technical noise removal
    let mut technical_component: Option<Vec<f64>> = None;
    let mut cellwise_bg: Option<Vec<f64>> = None;

    if params.denoise_counts {
        if verbose {
            println!("DSB: Step II - estimating per-cell background");
        }
        let bg = cellwise_background_means(&cell_log, params.kmeans_iters, seed);

        let noise_vec = if params.use_isotype_controls && !params.isotype_indices.is_empty() {
            let noise_rows = build_noise_matrix(&cell_log, &params.isotype_indices, &bg);
            pc1_scores(&noise_rows)?
        } else {
            // Match Seurat dsb fallback: regress out the per-cell background mean directly.
            bg.clone()
        };

        regress_out_covariate(&mut cell_log, &noise_vec);

        technical_component = Some(noise_vec);
        cellwise_bg = Some(bg);
    }

    // Optional quantile clipping per protein column
    if let Some((q_low, q_high)) = params.quantile_clip {
        if !(0.0..=1.0).contains(&q_low) || !(0.0..=1.0).contains(&q_high) || q_low >= q_high {
            return Err(BixverseErrors::InvalidArgument(format!(
                "invalid quantile_clip ({q_low}, {q_high})"
            )));
        }
        for j in 0..n_proteins {
            let ql = col_quantile(&cell_log, j, q_low);
            let qh = col_quantile(&cell_log, j, q_high);
            cell_log.par_iter_mut().for_each(|row| {
                if row[j] < ql {
                    row[j] = ql;
                } else if row[j] > qh {
                    row[j] = qh;
                }
            });
        }
    }

    // Convert back to f32
    let normalised = Mat::<f32>::from_fn(n_cells, n_proteins, |i, j| cell_log[i][j] as f32);

    Ok(DsbResult {
        normalised,
        protein_background_mean: bg_mean,
        protein_background_sd: bg_sd,
        technical_component,
        cellwise_background_mean: cellwise_bg,
    })
}
