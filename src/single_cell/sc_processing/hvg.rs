//! Implementations of highly variable gene detections in single cell. These
//! are based on the Seurat versions.

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::time::Instant;
use thousands::Separable;

use crate::core::base::info::{BinningStrategy, parse_bin_strategy_type};
use crate::core::base::loess::*;
use crate::prelude::*;

/////////
// HVG //
/////////

/// Structure that stores HVG information from VST
#[derive(Clone, Debug)]
pub struct HvgRes {
    /// Mean expression of the gene.
    pub mean: Vec<f64>,
    /// Detected variance of the gene.
    pub var: Vec<f64>,
    /// Expected variance of the gene.
    pub var_exp: Vec<f64>,
    /// Standardised variance of the gene.
    pub var_std: Vec<f64>,
}

/// Result structure for dispersion / mean-variance-bin HVG selection
#[derive(Clone, Debug)]
pub struct HvgDispersionRes {
    /// ExpMean: log1p(mean(expm1(data_norm))) per gene
    pub mean: Vec<f64>,
    /// LogVMR: log(var(expm1(data_norm)) / mean(expm1(data_norm))) per gene
    pub dispersion: Vec<f64>,
    /// Dispersion z-scored within mean-bin; 0 for genes outside a bin
    pub dispersion_scaled: Vec<f64>,
    /// Bin assignment per gene; -1 for genes not binned (constant / NaN)
    pub bin: Vec<i32>,
}

/// Enum for the different methods
#[derive(Debug, Clone, Copy, Default)]
pub enum HvgMethod {
    /// Variance stabilising transformation
    #[default]
    Vst,
    /// Binned version by average expression
    MeanVarBin,
    /// Simple dispersion
    Dispersion,
}

/// Helper function to parse the HVG
///
/// ### Params
///
/// * `s` - Type of HVG calculation to do
///
/// ### Returns
///
/// Option of the HvgMethod (some not yet implemented)
pub fn parse_hvg_method(s: &str) -> Option<HvgMethod> {
    match s.to_lowercase().as_str() {
        "vst" => Some(HvgMethod::Vst),
        "meanvarbin" => Some(HvgMethod::MeanVarBin),
        "dispersion" => Some(HvgMethod::Dispersion),
        _ => None,
    }
}

/////////////
// Helpers //
/////////////

/// Compute Seurat-style exp_mean and log_vmr on a gene chunk (uses data_norm)
///
/// Zero cells contribute 0 via expm1(0)=0, so iterating over nonzero entries
/// suffices. Sample variance uses (n-1) denominator to match R's var().
///
/// ### Params
///
/// * `chunk` - The `CscGeneChunk` representing the gene for which to calculate
///   the mean and variance
/// * `cell_idx_map` - Batch to cell mapping.
/// * `no_cells` - Number of total cells represented in the experiment
///
/// ### Returns
///
/// `(exp_mean, log_vmr)`. log_vmr is NaN for constant / all-zero genes.
#[inline]
pub fn calculate_disp_stats_filtered(
    gene: &CscGeneChunk,
    cell_idx_map: &FxHashMap<u32, u32>,
    no_cells: usize,
) -> (f32, f32) {
    let n = no_cells as f64;
    let mut sum = 0f64;
    let mut sum_sq = 0f64;

    for i in 0..gene.indices.len() {
        if cell_idx_map.contains_key(&gene.indices[i]) {
            let v = (gene.data_norm[i].to_f32() as f64).exp_m1();
            sum += v;
            sum_sq += v * v;
        }
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

    (exp_mean, log_vmr)
}

/// Assign genes to bins based on their mean expression
///
/// ### Params
///
/// * `means` - The mean values
/// * `method` - The binning strategy to apply
/// * `n_bins` - The number of bins to use
///
/// ### Returns
///
/// To which bin the given gene belongs
fn bin_features(means: &[f32], method: BinningStrategy, n_bins: usize) -> Vec<i32> {
    let n = means.len();
    let valid: Vec<bool> = means.iter().map(|v| v.is_finite()).collect();

    match method {
        BinningStrategy::EqualWidth => {
            let (mut vmin, mut vmax) = (f32::INFINITY, f32::NEG_INFINITY);
            for i in 0..n {
                if valid[i] {
                    if means[i] < vmin {
                        vmin = means[i];
                    }
                    if means[i] > vmax {
                        vmax = means[i];
                    }
                }
            }
            if !vmin.is_finite() || !vmax.is_finite() || vmax <= vmin {
                return vec![-1; n];
            }
            let width = (vmax - vmin) / n_bins as f32;
            (0..n)
                .map(|i| {
                    if !valid[i] {
                        -1
                    } else {
                        let idx = ((means[i] - vmin) / width) as i32;
                        idx.clamp(0, n_bins as i32 - 1)
                    }
                })
                .collect()
        }
        BinningStrategy::EqualFrequency => {
            let mut nonzero: Vec<f32> = means
                .iter()
                .zip(valid.iter())
                .filter_map(|(&v, &ok)| if ok && v > 0.0 { Some(v) } else { None })
                .collect();
            if nonzero.len() < 2 {
                return vec![-1; n];
            }
            nonzero.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let last = nonzero.len() - 1;
            let breaks: Vec<f32> = (0..=n_bins)
                .map(|i| {
                    let p = i as f32 / n_bins as f32;
                    let idx = (p * last as f32).round() as usize;
                    nonzero[idx.min(last)]
                })
                .collect();

            (0..n)
                .map(|i| {
                    if !valid[i] {
                        return -1;
                    }
                    let v = means[i];
                    if v < breaks[0] {
                        return 0;
                    }
                    for b in 0..n_bins {
                        if v <= breaks[b + 1] {
                            return b as i32;
                        }
                    }
                    (n_bins - 1) as i32
                })
                .collect()
        }
    }
}

/// Compute within-bin z-score of dispersion
///
/// ### Params
///
/// * `dispersion` - The dispersions
/// * `bins` - The bin assignments
/// * `n_bins` - The number of bins
///
/// ### Returns
///
/// Z-scores per bin
fn scale_within_bins(dispersion: &[f32], bins: &[i32], n_bins: usize) -> Vec<f32> {
    let mut sums = vec![0f64; n_bins];
    let mut sums_sq = vec![0f64; n_bins];
    let mut counts = vec![0usize; n_bins];

    for (&d, &b) in dispersion.iter().zip(bins.iter()) {
        if b >= 0 && d.is_finite() {
            let bi = b as usize;
            let df = d as f64;
            sums[bi] += df;
            sums_sq[bi] += df * df;
            counts[bi] += 1;
        }
    }

    let stats: Vec<(f64, f64)> = (0..n_bins)
        .map(|i| {
            if counts[i] < 2 {
                (0.0, 0.0)
            } else {
                let c = counts[i] as f64;
                let m = sums[i] / c;
                let var = ((sums_sq[i] - c * m * m) / (c - 1.0)).max(0.0);
                (m, var.sqrt())
            }
        })
        .collect();

    dispersion
        .iter()
        .zip(bins.iter())
        .map(|(&d, &b)| {
            if b < 0 || !d.is_finite() {
                return 0.0;
            }
            let (m, sd) = stats[b as usize];
            if sd > 0.0 {
                ((d as f64 - m) / sd) as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// Build the final HvgDispersionRes from raw per-gene means and dispersions
///
/// ### Params
///
/// * `means` - The mean expression of the gene
/// * `dispersions` - The dispersions of the gene
/// * `binning` - The binning strategy
/// * `n_bins` - Number of bins
///
/// ### Returns
///
/// The `HvgDispersionRes`
pub fn build_disp_result(
    means: Vec<f32>,
    dispersions: Vec<f32>,
    binning: BinningStrategy,
    n_bins: usize,
) -> HvgDispersionRes {
    let bins = bin_features(&means, binning, n_bins);
    let scaled = scale_within_bins(&dispersions, &bins, n_bins);

    // replace NaN in means / dispersion with 0 (matches Seurat)
    let means_clean: Vec<f32> = means
        .iter()
        .map(|&v| if v.is_finite() { v } else { 0.0 })
        .collect();
    let disp_clean: Vec<f32> = dispersions
        .iter()
        .map(|&v| if v.is_finite() { v } else { 0.0 })
        .collect();

    HvgDispersionRes {
        mean: means_clean.r_float_convert(),
        dispersion: disp_clean.r_float_convert(),
        dispersion_scaled: scaled.r_float_convert(),
        bin: bins,
    }
}

/// Calculate the mean and variance of a CSC gene chunk
///
/// ### Params
///
/// * `chunk` - The `CscGeneChunk` representing the gene for which to calculate
///   the mean and variance
/// * `cell_idx_map` - Batch to cell mapping.
/// * `no_cells` - Number of total cells represented in the experiment
///
/// ### Returns
///
/// A tuple of `(mean, var)`
#[inline]
pub fn calculate_mean_var_filtered(
    gene: &CscGeneChunk,
    cell_idx_map: &FxHashMap<u32, u32>,
    no_cells: usize,
) -> (f32, f32) {
    let no_cells = no_cells as f32;
    let mut sum = 0f32;
    let mut nnz = 0usize;

    // only process cells that are in the filter
    for i in 0..gene.indices.len() {
        if cell_idx_map.contains_key(&gene.indices[i]) {
            sum += gene.data_raw.get(i) as f32;
            nnz += 1;
        }
    }

    let n_zero = no_cells - nnz as f32;
    let mean = sum / no_cells;

    let mut sum_sq_diff = 0f32;
    for i in 0..gene.indices.len() {
        if cell_idx_map.contains_key(&gene.indices[i]) {
            let val = gene.data_raw.get(i) as f32;
            let diff = val - mean;
            sum_sq_diff += diff * diff;
        }
    }

    let var = (sum_sq_diff + n_zero * mean * mean) / no_cells;

    (mean, var)
}

/// Helper function to calculate the standardised variance of a gene chunk
///
/// ### Params
///
/// * `chunk` - The `CscGeneChunk` representing the gene for which to calculate
///   the standardised mean and variance
/// * `mean` - Mean value for that gene
/// * `expected_var` - Expected variance based on the Loess function
/// * `clip_max` - Which values to clip
/// * `no_cells` - The number of represented cells
///
/// ### Returns
///
/// The standardised variance
#[inline]
pub fn calculate_std_variance_filtered(
    gene: &CscGeneChunk,
    cell_idx_map: &FxHashMap<u32, u32>,
    mean: f32,
    expected_var: f32,
    clip_max: f32,
    no_cells: usize,
) -> f32 {
    let no_cells_f32 = no_cells as f32;
    let expected_sd = expected_var.sqrt();

    let mut sum_standardised = 0f32;
    let mut sum_sq_standardised = 0f32;
    let mut nnz = 0usize;

    // Process non-zero entries that pass filter
    for i in 0..gene.indices.len() {
        if cell_idx_map.contains_key(&gene.indices[i]) {
            let val_f32 = gene.data_raw.get(i) as f32;
            let norm = ((val_f32 - mean) / expected_sd)
                .min(clip_max)
                .max(-clip_max);
            sum_standardised += norm;
            sum_sq_standardised += norm * norm;
            nnz += 1;
        }
    }

    // Process zero entries
    let n_zeros = no_cells - nnz;
    if n_zeros > 0 {
        let standardised_zero = ((-mean) / expected_sd).min(clip_max).max(-clip_max);
        sum_standardised += n_zeros as f32 * standardised_zero;
        sum_sq_standardised += n_zeros as f32 * standardised_zero * standardised_zero;
    }

    let standardised_mean = sum_standardised / no_cells_f32;
    (sum_sq_standardised / no_cells_f32) - (standardised_mean * standardised_mean)
}

/////////
// VST //
/////////

/// Implementation of the variance stabilised version of the HVG selection
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - HashSet with the cell indices to keep.
/// * `loess_span` - Span parameter for the loess function
/// * `clip_max` - Optional clip max parameter
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The `HvgRes`
pub fn get_hvg_vst<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    loess_span: f32,
    clip_max: Option<f32>,
    verbose: usize,
) -> Result<HvgRes, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    let start_total = Instant::now();

    // Get data
    let start_read = Instant::now();

    let mut gene_chunks: Vec<CscGeneChunk> = reader.get_all_genes()?;
    let no_cells = cell_indices.len();

    // build cell mapping ONCE... Before I was doing stupid shit
    let cell_idx_map: FxHashMap<u32, u32> = cell_indices
        .iter()
        .enumerate()
        .map(|(new_idx, &old_idx)| (old_idx as u32, new_idx as u32))
        .collect();

    let end_read = start_read.elapsed();

    if verbosity.normal_verbosity() {
        println!("HVG (VST): Load in data in {:.2?}", end_read);
    }

    let start_gene_stats = Instant::now();

    let results: Vec<(f32, f32)> = gene_chunks
        .par_iter_mut()
        .map(|chunk| calculate_mean_var_filtered(chunk, &cell_idx_map, no_cells))
        .collect();

    let end_gene_stats = start_gene_stats.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "HVG (VST): Calculated gene statistics in {:.2?}",
            end_gene_stats
        );
    }

    let start_loess = Instant::now();

    let (means, vars): (Vec<f32>, Vec<f32>) = results.into_iter().unzip();

    let clip_max = clip_max.unwrap_or((no_cells as f32).sqrt());
    let means_log10: Vec<f32> = means.iter().map(|x| x.log10()).collect();
    let vars_log10: Vec<f32> = vars.iter().map(|x| x.log10()).collect();

    let loess = LoessRegression::new(loess_span, 2);
    let loess_res = loess.fit(&means_log10, &vars_log10);

    let end_loess = start_loess.elapsed();

    if verbosity.normal_verbosity() {
        println!("HVG (VST): Fitted Loess in {:.2?}", end_loess);
    }

    let start_standard = Instant::now();

    let var_standardised: Vec<f32> = gene_chunks
        .par_iter()
        .zip(loess_res.fitted_vals.par_iter())
        .zip(means.par_iter())
        .map(|((chunk_i, var_i), mean_i)| {
            let expected_var = 10_f32.powf(*var_i);
            calculate_std_variance_filtered(
                chunk_i,
                &cell_idx_map,
                *mean_i,
                expected_var,
                clip_max,
                no_cells,
            )
        })
        .collect();

    let end_standard = start_standard.elapsed();

    if verbosity.normal_verbosity() {
        println!("HVG (VST): Standardised variance in {:.2?}", end_standard);
    }

    let total = start_total.elapsed();

    if verbosity.normal_verbosity() {
        println!("HVG (VST): Total run time -> {:.2?}", total);
    }

    // transform to f64 for R
    Ok(HvgRes {
        mean: means.r_float_convert(),
        var: vars.r_float_convert(),
        var_exp: loess_res.fitted_vals.r_float_convert(),
        var_std: var_standardised.r_float_convert(),
    })
}

/// Implementation of the variance stabilised version of the HVG selection
///
/// This uses a two-pass approach to minimise memory usage:
/// - Pass 1: Calculate mean/variance for loess fitting (genes processed in
///   batches)
/// - Pass 2: Calculate standardized variance using loess results (genes
///   processed in batches)
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep.
/// * `loess_span` - Span parameter for the loess function
/// * `clip_max` - Optional clip max parameter
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The `HvgRes`
pub fn get_hvg_vst_streaming<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    loess_span: f32,
    clip_max: Option<f32>,
    verbose: usize,
) -> Result<HvgRes, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    let start_total = Instant::now();

    let header = reader.get_header();
    let no_genes = header.total_genes;
    let no_cells = cell_indices.len();

    let cell_idx_map: FxHashMap<u32, u32> = cell_indices
        .iter()
        .enumerate()
        .map(|(new_idx, &old_idx)| (old_idx as u32, new_idx as u32))
        .collect();

    if verbosity.normal_verbosity() {
        println!(
            "HVG (VST, streaming): Pass 1/2 -> Calculating mean and variance for {} genes...",
            no_genes.separate_with_underscores()
        );
    }

    // first pass -> calculate mean and variance in batches
    let start_pass1 = Instant::now();

    const GENE_BATCH_SIZE: usize = 1000;
    let num_batches = no_genes.div_ceil(GENE_BATCH_SIZE);

    let mut means = Vec::with_capacity(no_genes);
    let mut vars = Vec::with_capacity(no_genes);

    for batch_idx in 0..num_batches {
        if verbosity.detailed_verbosity() && batch_idx % 5 == 0 {
            let progress = (batch_idx + 1) as f32 / num_batches as f32 * 100.0;
            println!("  Progress: {:.1}%", progress);
        }

        let start_gene = batch_idx * GENE_BATCH_SIZE;
        let end_gene = ((batch_idx + 1) * GENE_BATCH_SIZE).min(no_genes);
        let gene_indices: Vec<usize> = (start_gene..end_gene).collect();

        let start_loading = Instant::now();

        let mut genes = reader.read_gene_parallel(&gene_indices)?;

        let end_loading = start_loading.elapsed();

        if verbosity.detailed_verbosity() {
            println!("   Loaded batch in: {:.2?}.", end_loading);
        }

        let start_batch = Instant::now();

        let batch_results: Vec<(f32, f32)> = genes
            .par_iter_mut()
            .map(|gene| calculate_mean_var_filtered(gene, &cell_idx_map, no_cells))
            .collect();

        let end_batch = start_batch.elapsed();

        if verbosity.detailed_verbosity() {
            println!("   Finished calculations in: {:.2?}.", end_batch);
        }

        for (mean, var) in batch_results {
            means.push(mean);
            vars.push(var);
        }

        drop(genes)
    }

    let end_pass1 = start_pass1.elapsed();

    if verbosity.normal_verbosity() {
        println!("  Calculated gene statistics: {:.2?}", end_pass1);
    }

    // Fit loess
    let start_loess = Instant::now();

    let clip_max = clip_max.unwrap_or((no_cells as f32).sqrt());
    let means_log10: Vec<f32> = means.iter().map(|x| x.log10()).collect();
    let vars_log10: Vec<f32> = vars.iter().map(|x| x.log10()).collect();

    let loess = LoessRegression::new(loess_span, 2);
    let loess_res = loess.fit(&means_log10, &vars_log10);

    let end_loess = start_loess.elapsed();

    if verbosity.normal_verbosity() {
        println!("  Fitted Loess: {:.2?}", end_loess);
        println!("HVG (VST, streaming): Pass 2/2 -> Calculating standardised variance...");
    }

    // second pass -> calculate standardised variance in batches
    let start_pass2 = Instant::now();

    let mut var_standardised = Vec::with_capacity(no_genes);

    for batch_idx in 0..num_batches {
        if verbosity.detailed_verbosity() && batch_idx % 5 == 0 {
            let progress = (batch_idx + 1) as f32 / num_batches as f32 * 100.0;
            println!("  Progress: {:.1}%", progress);
        }

        let start_gene = batch_idx * GENE_BATCH_SIZE;
        let end_gene = ((batch_idx + 1) * GENE_BATCH_SIZE).min(no_genes);
        let gene_indices: Vec<usize> = (start_gene..end_gene).collect();

        let mut genes = reader.read_gene_parallel(&gene_indices)?;

        let start_batch = Instant::now();

        let batch_std_vars: Vec<f32> = genes
            .par_iter_mut()
            .enumerate()
            .map(|(local_idx, gene)| {
                let gene_idx = start_gene + local_idx;
                let expected_var = 10_f32.powf(loess_res.fitted_vals[gene_idx]);
                calculate_std_variance_filtered(
                    gene,
                    &cell_idx_map,
                    means[gene_idx],
                    expected_var,
                    clip_max,
                    no_cells,
                )
            })
            .collect();

        let end_batch = start_batch.elapsed();

        if verbosity.detailed_verbosity() {
            println!(
                "   Finished calculating standardised variance in: {:.2?}.",
                end_batch
            );
        }

        drop(genes);

        var_standardised.extend(batch_std_vars);
    }

    let end_pass2 = start_pass2.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "  Calculated standardised variance total: {:.2?}",
            end_pass2
        );
    }

    let total = start_total.elapsed();

    if verbosity.normal_verbosity() {
        println!("HVG (VST, streaming): Total run time -> {:.2?}", total);
    }

    // transform to f64 for R
    Ok(HvgRes {
        mean: means.r_float_convert(),
        var: vars.r_float_convert(),
        var_exp: loess_res.fitted_vals.r_float_convert(),
        var_std: var_standardised.r_float_convert(),
    })
}

/////////////////////////
// Dispersion versions //
/////////////////////////

/// Dispersion-based HVG detection (non-streaming)
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep.
/// * `binning` - The binning strategy to use. One of
///   `"equal_width"` or `"equal_freq"`
/// * `n_bins` - The number of bins to use
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The `HvgDispersionRes`
pub fn get_hvg_dispersion<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    binning: &str,
    n_bins: usize,
    verbose: usize,
) -> Result<HvgDispersionRes, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    let start_total = Instant::now();

    let binning = parse_bin_strategy_type(binning).unwrap_or_default();

    let start_read = Instant::now();
    let mut gene_chunks: Vec<CscGeneChunk> = reader.get_all_genes()?;
    let no_cells = cell_indices.len();

    let cell_idx_map: FxHashMap<u32, u32> = cell_indices
        .iter()
        .enumerate()
        .map(|(new_idx, &old_idx)| (old_idx as u32, new_idx as u32))
        .collect();

    if verbosity.normal_verbosity() {
        println!(
            "HVG (dispersion): Load in data in {:.2?}",
            start_read.elapsed()
        );
    }

    let start_stats = Instant::now();
    let results: Vec<(f32, f32)> = gene_chunks
        .par_iter_mut()
        .map(|chunk| calculate_disp_stats_filtered(chunk, &cell_idx_map, no_cells))
        .collect();

    if verbosity.normal_verbosity() {
        println!(
            "HVG (dispersion): Calculated gene statistics in {:.2?}",
            start_stats.elapsed()
        );
    }

    let (means, dispersions): (Vec<f32>, Vec<f32>) = results.into_iter().unzip();

    let start_bin = Instant::now();
    let res = build_disp_result(means, dispersions, binning, n_bins);
    if verbosity.normal_verbosity() {
        println!(
            "HVG (dispersion): Binning and scaling in {:.2?}",
            start_bin.elapsed()
        );
        println!(
            "HVG (dispersion): Total run time -> {:.2?}",
            start_total.elapsed()
        );
    }

    Ok(res)
}

/// Dispersion-based HVG detection (streaming)
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep.
/// * `binning` - The binning strategy to use. One of
///   `"equal_width"` or `"equal_freq"`
/// * `n_bins` - The number of bins to use
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The `HvgDispersionRes`
pub fn get_hvg_dispersion_streaming<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    binning: &str,
    n_bins: usize,
    verbose: usize,
) -> Result<HvgDispersionRes, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    let start_total = Instant::now();

    let binning = parse_bin_strategy_type(binning).unwrap_or_default();

    let header = reader.get_header();
    let no_genes = header.total_genes;
    let no_cells = cell_indices.len();

    let cell_idx_map: FxHashMap<u32, u32> = cell_indices
        .iter()
        .enumerate()
        .map(|(new_idx, &old_idx)| (old_idx as u32, new_idx as u32))
        .collect();

    if verbosity.normal_verbosity() {
        println!(
            "HVG (dispersion, streaming): Calculating dispersion stats for {} genes...",
            no_genes.separate_with_underscores()
        );
    }

    const GENE_BATCH_SIZE: usize = 1000;
    let num_batches = no_genes.div_ceil(GENE_BATCH_SIZE);

    let mut means = Vec::with_capacity(no_genes);
    let mut dispersions = Vec::with_capacity(no_genes);

    for batch_idx in 0..num_batches {
        if verbosity.detailed_verbosity() && batch_idx % 5 == 0 {
            let progress = (batch_idx + 1) as f32 / num_batches as f32 * 100.0;
            println!("  Progress: {:.1}%", progress);
        }

        let start_gene = batch_idx * GENE_BATCH_SIZE;
        let end_gene = ((batch_idx + 1) * GENE_BATCH_SIZE).min(no_genes);
        let gene_indices: Vec<usize> = (start_gene..end_gene).collect();

        let mut genes = reader.read_gene_parallel(&gene_indices)?;

        let batch_results: Vec<(f32, f32)> = genes
            .par_iter_mut()
            .map(|gene| calculate_disp_stats_filtered(gene, &cell_idx_map, no_cells))
            .collect();

        for (m, d) in batch_results {
            means.push(m);
            dispersions.push(d);
        }

        drop(genes);
    }

    let start_bin = Instant::now();
    let res = build_disp_result(means, dispersions, binning, n_bins);

    if verbosity.normal_verbosity() {
        println!(
            "HVG (dispersion, streaming): Binning and scaling in {:.2?}",
            start_bin.elapsed()
        );
        println!(
            "HVG (dispersion, streaming): Total run time -> {:.2?}",
            start_total.elapsed()
        );
    }

    Ok(res)
}

/// MVB is computationally identical to dispersion
///
/// Selection differs on the R side.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep.
/// * `binning` - The binning strategy to use. One of
///   `"equal_width"` or `"equal_freq"`
/// * `n_bins` - The number of bins to use
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The `HvgDispersionRes`
pub fn get_hvg_mvb<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    binning: &str,
    n_bins: usize,
    verbose: usize,
) -> Result<HvgDispersionRes, BixverseErrors> {
    get_hvg_dispersion(reader, cell_indices, binning, n_bins, verbose)
}

/// MVB is computationally identical to dispersion (streaming)
///
/// Selection differs on the R side.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep.
/// * `binning` - The binning strategy to use. One of
///   `"equal_width"` or `"equal_freq"`
/// * `n_bins` - The number of bins to use
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The `HvgDispersionRes`
pub fn get_hvg_mvb_streaming<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    binning: &str,
    n_bins: usize,
    verbose: usize,
) -> Result<HvgDispersionRes, BixverseErrors> {
    get_hvg_dispersion_streaming(reader, cell_indices, binning, n_bins, verbose)
}

/////////////////////
// HVG batch aware //
/////////////////////

/////////
// VST //
/////////

/// Batch-aware HVG selection using VST method
///
/// Calculates HVG statistics separately for each batch, returning per-batch results.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep
/// * `batch_labels` - Batch assignment for each cell (same length as cell_indices)
/// * `loess_span` - Span parameter for the loess function
/// * `clip_max` - Optional clip max parameter
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `Vec<HvgRes>` - One HvgRes per batch
pub fn get_hvg_vst_batch_aware<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    batch_labels: &[usize],
    loess_span: f32,
    clip_max: Option<f32>,
    verbose: usize,
) -> Result<Vec<HvgRes>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    let start_total = Instant::now();

    // batch cell maps
    let start_setup = Instant::now();

    let n_batches = *batch_labels.iter().max().unwrap() + 1;
    let mut batch_cell_maps: Vec<FxHashMap<u32, u32>> = vec![FxHashMap::default(); n_batches];
    let mut batch_sizes = vec![0usize; n_batches];

    for (&old_idx, &batch) in cell_indices.iter().zip(batch_labels.iter()) {
        batch_cell_maps[batch].insert(old_idx as u32, batch_sizes[batch] as u32);
        batch_sizes[batch] += 1;
    }

    let end_setup = start_setup.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "HVG (VST, batch-aware): Setup batch maps in {:.2?}",
            end_setup
        );
        println!(" Processing {} batches", n_batches);
    }

    // load data
    let start_read = Instant::now();
    let mut gene_chunks: Vec<CscGeneChunk> = reader.get_all_genes()?;
    let end_read = start_read.elapsed();

    if verbosity.normal_verbosity() {
        println!("HVG (VST, batch-aware): Loaded data in {:.2?}", end_read);
    }

    // Calculate the gene statistics
    let start_pass_1 = Instant::now();

    let mut batch_means: Vec<Vec<f32>> = vec![Vec::new(); n_batches];
    let mut batch_vars: Vec<Vec<f32>> = vec![Vec::new(); n_batches];

    for batch_idx in 0..n_batches {
        let results: Vec<(f32, f32)> = gene_chunks
            .par_iter_mut()
            .map(|chunk| {
                calculate_mean_var_filtered(
                    chunk,
                    &batch_cell_maps[batch_idx],
                    batch_sizes[batch_idx],
                )
            })
            .collect();

        let (means, vars): (Vec<f32>, Vec<f32>) = results.into_iter().unzip();
        batch_means[batch_idx] = means;
        batch_vars[batch_idx] = vars;
    }

    let end_pass_1 = start_pass_1.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "HVG (VST, batch-aware): Calculated gene statistics per batch in {:.2?}",
            end_pass_1
        );
    }

    // fit loess per batch
    let start_loess = Instant::now();

    let mut batch_loess_results = Vec::with_capacity(n_batches);
    let mut batch_clip_max = Vec::with_capacity(n_batches);

    for batch_idx in 0..n_batches {
        let clip = clip_max.unwrap_or((batch_sizes[batch_idx] as f32).sqrt());
        batch_clip_max.push(clip);

        let means_log10: Vec<f32> = batch_means[batch_idx].iter().map(|x| x.log10()).collect();
        let vars_log10: Vec<f32> = batch_vars[batch_idx].iter().map(|x| x.log10()).collect();

        let loess = LoessRegression::new(loess_span, 2);
        let loess_res = loess.fit(&means_log10, &vars_log10);
        batch_loess_results.push(loess_res);
    }

    let end_loess = start_loess.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "HVG (VST, batch-aware): Fitted loess per batch in {:.2?}",
            end_loess
        );
    }

    // standardise variance
    let start_pass_2 = Instant::now();

    let mut batch_std_vars: Vec<Vec<f32>> = vec![Vec::new(); n_batches];

    for batch_idx in 0..n_batches {
        let var_standardised: Vec<f32> = gene_chunks
            .par_iter()
            .enumerate()
            .map(|(gene_idx, chunk)| {
                let expected_var =
                    10_f32.powf(batch_loess_results[batch_idx].fitted_vals[gene_idx]);
                calculate_std_variance_filtered(
                    chunk,
                    &batch_cell_maps[batch_idx],
                    batch_means[batch_idx][gene_idx],
                    expected_var,
                    batch_clip_max[batch_idx],
                    batch_sizes[batch_idx],
                )
            })
            .collect();

        batch_std_vars[batch_idx] = var_standardised;
    }

    let end_pass_2 = start_pass_2.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "HVG (VST, batch-aware): Calculated standardised variance per batch in {:.2?}",
            end_pass_2
        );
    }

    let total = start_total.elapsed();

    if verbosity.normal_verbosity() {
        println!("HVG (VST, batch-aware): Total runtime -> {:.2?}", total);
    }

    let res = batch_means
        .into_iter()
        .zip(batch_vars)
        .zip(batch_loess_results)
        .zip(batch_std_vars)
        .map(|(((means, vars), loess_res), std_vars)| HvgRes {
            mean: means.r_float_convert(),
            var: vars.r_float_convert(),
            var_exp: loess_res.fitted_vals.r_float_convert(),
            var_std: std_vars.r_float_convert(),
        })
        .collect();

    Ok(res)
}

/// Batch-aware HVG selection using VST method with streaming
///
/// Two-pass approach processing genes in batches to minimise memory usage.
/// Calculates HVG statistics separately for each batch.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep
/// * `batch_labels` - Batch assignment for each cell (same length as cell_
///   indices)
/// * `loess_span` - Span parameter for the loess function
/// * `clip_max` - Optional clip max parameter
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `Vec<HvgRes>` - One HvgRes per batch
pub fn get_hvg_vst_batch_aware_streaming<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    batch_labels: &[usize],
    loess_span: f32,
    clip_max: Option<f32>,
    verbose: usize,
) -> Result<Vec<HvgRes>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    let start_total = Instant::now();

    let header = reader.get_header();
    let no_genes = header.total_genes;

    // Build batch cell maps
    let start_setup = Instant::now();

    let n_batches = *batch_labels.iter().max().unwrap() + 1;
    let mut batch_cell_maps: Vec<FxHashMap<u32, u32>> = vec![FxHashMap::default(); n_batches];
    let mut batch_sizes = vec![0usize; n_batches];

    for (&old_idx, &batch) in cell_indices.iter().zip(batch_labels.iter()) {
        batch_cell_maps[batch].insert(old_idx as u32, batch_sizes[batch] as u32);
        batch_sizes[batch] += 1;
    }

    let end_setup = start_setup.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "HVG (VST, streaming, batch-aware): Setup batch maps in {:.2?}",
            end_setup
        );
        println!(" Processing {} batches", n_batches);
    }

    // Pass 1: Calculate mean and variance per batch
    if verbosity.normal_verbosity() {
        println!(
            "HVG (VST, streaming, batch-aware): Pass 1/2 -> Calculating mean and variance for {} genes...",
            no_genes.separate_with_underscores()
        );
    }

    let start_pass1 = Instant::now();
    const GENE_BATCH_SIZE: usize = 1000;
    let num_batches = no_genes.div_ceil(GENE_BATCH_SIZE);

    let mut batch_means: Vec<Vec<f32>> = vec![Vec::with_capacity(no_genes); n_batches];
    let mut batch_vars: Vec<Vec<f32>> = vec![Vec::with_capacity(no_genes); n_batches];

    for chunk_idx in 0..num_batches {
        if verbosity.detailed_verbosity() && chunk_idx % 5 == 0 {
            let progress = (chunk_idx + 1) as f32 / num_batches as f32 * 100.0;
            println!("  Progress: {:.1}%", progress);
        }

        let start_gene = chunk_idx * GENE_BATCH_SIZE;
        let end_gene = ((chunk_idx + 1) * GENE_BATCH_SIZE).min(no_genes);
        let gene_indices: Vec<usize> = (start_gene..end_gene).collect();

        let mut genes = reader.read_gene_parallel(&gene_indices)?;

        // Calculate mean/var for all batches on this gene chunk
        for batch_idx in 0..n_batches {
            let results: Vec<(f32, f32)> = genes
                .par_iter_mut()
                .map(|gene| {
                    calculate_mean_var_filtered(
                        gene,
                        &batch_cell_maps[batch_idx],
                        batch_sizes[batch_idx],
                    )
                })
                .collect();

            for (mean, var) in results {
                batch_means[batch_idx].push(mean);
                batch_vars[batch_idx].push(var);
            }
        }
    }

    let end_pass1 = start_pass1.elapsed();

    if verbosity.normal_verbosity() {
        println!("  Calculated gene statistics per batch: {:.2?}", end_pass1);
    }

    // Fit loess per batch
    let start_loess = Instant::now();

    let mut batch_loess_results = Vec::with_capacity(n_batches);
    let mut batch_clip_max = Vec::with_capacity(n_batches);

    for batch_idx in 0..n_batches {
        let clip = clip_max.unwrap_or((batch_sizes[batch_idx] as f32).sqrt());
        batch_clip_max.push(clip);

        let means_log10: Vec<f32> = batch_means[batch_idx].iter().map(|x| x.log10()).collect();
        let vars_log10: Vec<f32> = batch_vars[batch_idx].iter().map(|x| x.log10()).collect();

        let loess = LoessRegression::new(loess_span, 2);
        let loess_res = loess.fit(&means_log10, &vars_log10);
        batch_loess_results.push(loess_res);
    }

    let end_loess = start_loess.elapsed();

    if verbosity.normal_verbosity() {
        println!("  Fitted loess per batch: {:.2?}", end_loess);
        println!(
            "HVG (VST, streaming, batch-aware): Pass 2/2 -> Calculating standardised variance..."
        );
    }

    // Pass 2: Calculate standardised variance per batch
    let start_pass2 = Instant::now();

    let mut batch_std_vars: Vec<Vec<f32>> = vec![Vec::with_capacity(no_genes); n_batches];

    for chunk_idx in 0..num_batches {
        if verbosity.detailed_verbosity() && chunk_idx % 5 == 0 {
            let progress = (chunk_idx + 1) as f32 / num_batches as f32 * 100.0;
            println!("  Progress: {:.1}%", progress);
        }

        let start_gene = chunk_idx * GENE_BATCH_SIZE;
        let end_gene = ((chunk_idx + 1) * GENE_BATCH_SIZE).min(no_genes);
        let gene_indices: Vec<usize> = (start_gene..end_gene).collect();

        let mut genes = reader.read_gene_parallel(&gene_indices)?;

        // Calculate std_var for all batches on this gene chunk
        for batch_idx in 0..n_batches {
            let std_vars: Vec<f32> = genes
                .par_iter_mut()
                .enumerate()
                .map(|(local_idx, gene)| {
                    let gene_idx = start_gene + local_idx;
                    let expected_var =
                        10_f32.powf(batch_loess_results[batch_idx].fitted_vals[gene_idx]);
                    calculate_std_variance_filtered(
                        gene,
                        &batch_cell_maps[batch_idx],
                        batch_means[batch_idx][gene_idx],
                        expected_var,
                        batch_clip_max[batch_idx],
                        batch_sizes[batch_idx],
                    )
                })
                .collect();

            batch_std_vars[batch_idx].extend(std_vars);
        }
    }

    let end_pass2 = start_pass2.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "  Calculated standardised variance per batch: {:.2?}",
            end_pass2
        );
    }

    let total = start_total.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "HVG (VST, streaming, batch-aware): Total runtime -> {:.2?}",
            total
        );
    }

    // Build results per batch
    let res = batch_means
        .into_iter()
        .zip(batch_vars)
        .zip(batch_loess_results)
        .zip(batch_std_vars)
        .map(|(((means, vars), loess_res), std_vars)| HvgRes {
            mean: means.r_float_convert(),
            var: vars.r_float_convert(),
            var_exp: loess_res.fitted_vals.r_float_convert(),
            var_std: std_vars.r_float_convert(),
        })
        .collect();

    Ok(res)
}

/////////////////////////
// Dispersion versions //
/////////////////////////

/// Dispersion-based HVG detection, batch-aware
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep
/// * `batch_labels` - Batch assignment for each cell (same length as cell_
///   indices)
/// * `binning` - The binning strategy to use. One of
///   `"equal_width"` or `"equal_freq"`
/// * `n_bins` - Number of bins
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A `Vec<HvgDispersionRes>` with each element being a batch.
pub fn get_hvg_dispersion_batch_aware<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    batch_labels: &[usize],
    binning: &str,
    n_bins: usize,
    verbose: usize,
) -> Result<Vec<HvgDispersionRes>, BixverseErrors> {
    let start_total = Instant::now();

    let binning = parse_bin_strategy_type(binning).unwrap_or_default();
    let verbosity = parse_verbosity_level(verbose);

    let n_batches = *batch_labels.iter().max().unwrap() + 1;
    let mut batch_cell_maps: Vec<FxHashMap<u32, u32>> = vec![FxHashMap::default(); n_batches];
    let mut batch_sizes = vec![0usize; n_batches];

    for (&old_idx, &batch) in cell_indices.iter().zip(batch_labels.iter()) {
        batch_cell_maps[batch].insert(old_idx as u32, batch_sizes[batch] as u32);
        batch_sizes[batch] += 1;
    }

    if verbosity.normal_verbosity() {
        println!(
            "HVG (Dispersion, batch-aware): Processing {} batches",
            n_batches
        );
    }

    let start_read = Instant::now();
    let mut gene_chunks: Vec<CscGeneChunk> = reader.get_all_genes()?;
    if verbosity.normal_verbosity() {
        println!(
            "HVG (Dispersion, batch-aware): Loaded data in {:.2?}",
            start_read.elapsed()
        );
    }

    let mut out = Vec::with_capacity(n_batches);

    for batch_idx in 0..n_batches {
        let start_batch = Instant::now();
        let results: Vec<(f32, f32)> = gene_chunks
            .par_iter_mut()
            .map(|chunk| {
                calculate_disp_stats_filtered(
                    chunk,
                    &batch_cell_maps[batch_idx],
                    batch_sizes[batch_idx],
                )
            })
            .collect();

        let (means, dispersions): (Vec<f32>, Vec<f32>) = results.into_iter().unzip();
        let res = build_disp_result(means, dispersions, binning, n_bins);
        out.push(res);

        if verbosity.detailed_verbosity() {
            println!(
                " Batch {}/{}: {:.2?}",
                batch_idx + 1,
                n_batches,
                start_batch.elapsed()
            );
        }
    }

    if verbosity.normal_verbosity() {
        println!(
            "HVG (Dispersion, batch-aware): Total runtime -> {:.2?}",
            start_total.elapsed()
        );
    }

    Ok(out)
}

/// Dispersion-based HVG detection, batch-aware with streaming
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep
/// * `batch_labels` - Batch assignment for each cell (same length as cell_
///   indices)
/// * `binning` - The binning strategy to use. One of
///   `"equal_width"` or `"equal_freq"`
/// * `n_bins` - Number of bins
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A `Vec<HvgDispersionRes>` with each element being a batch.
pub fn get_hvg_dispersion_batch_aware_streaming<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    batch_labels: &[usize],
    binning: &str,
    n_bins: usize,
    verbose: usize,
) -> Result<Vec<HvgDispersionRes>, BixverseErrors> {
    let start_total = Instant::now();

    let verbosity = parse_verbosity_level(verbose);

    let binning = parse_bin_strategy_type(binning).unwrap_or_default();

    let header = reader.get_header();
    let no_genes = header.total_genes;

    let n_batches = *batch_labels.iter().max().unwrap() + 1;
    let mut batch_cell_maps: Vec<FxHashMap<u32, u32>> = vec![FxHashMap::default(); n_batches];
    let mut batch_sizes = vec![0usize; n_batches];

    for (&old_idx, &batch) in cell_indices.iter().zip(batch_labels.iter()) {
        batch_cell_maps[batch].insert(old_idx as u32, batch_sizes[batch] as u32);
        batch_sizes[batch] += 1;
    }

    if verbosity.normal_verbosity() {
        println!(
            "HVG (Dispersion, batch-aware, streaming): Processing {} batches",
            n_batches
        );
        println!(
            " Calculating dispersion stats for {} genes...",
            no_genes.separate_with_underscores()
        );
    }

    const GENE_BATCH_SIZE: usize = 1000;
    let num_batches = no_genes.div_ceil(GENE_BATCH_SIZE);

    let mut batch_means: Vec<Vec<f32>> = vec![Vec::with_capacity(no_genes); n_batches];
    let mut batch_dispersions: Vec<Vec<f32>> = vec![Vec::with_capacity(no_genes); n_batches];

    for chunk_idx in 0..num_batches {
        if verbosity.detailed_verbosity() && chunk_idx % 5 == 0 {
            let progress = (chunk_idx + 1) as f32 / num_batches as f32 * 100.0;
            println!("  Progress: {:.1}%", progress);
        }

        let start_gene = chunk_idx * GENE_BATCH_SIZE;
        let end_gene = ((chunk_idx + 1) * GENE_BATCH_SIZE).min(no_genes);
        let gene_indices: Vec<usize> = (start_gene..end_gene).collect();

        let mut genes = reader.read_gene_parallel(&gene_indices)?;

        for batch_idx in 0..n_batches {
            let results: Vec<(f32, f32)> = genes
                .par_iter_mut()
                .map(|gene| {
                    calculate_disp_stats_filtered(
                        gene,
                        &batch_cell_maps[batch_idx],
                        batch_sizes[batch_idx],
                    )
                })
                .collect();

            for (m, d) in results {
                batch_means[batch_idx].push(m);
                batch_dispersions[batch_idx].push(d);
            }
        }

        drop(genes);
    }

    let out: Vec<HvgDispersionRes> = batch_means
        .into_iter()
        .zip(batch_dispersions)
        .map(|(means, dispersions)| build_disp_result(means, dispersions, binning, n_bins))
        .collect();

    if verbosity.normal_verbosity() {
        println!(
            "HVG (Dispersion, batch-aware, streaming): Total runtime -> {:.2?}",
            start_total.elapsed()
        );
    }

    Ok(out)
}

/// MVB HVG, batch-aware
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep
/// * `batch_labels` - Batch assignment for each cell (same length as cell_
///   indices)
/// * `binning` - The binning strategy to use. One of
///   `"equal_width"` or `"equal_freq"`
/// * `n_bins` - Number of bins
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A `Vec<HvgDispersionRes>` with each element being a batch.
pub fn get_hvg_mvb_batch_aware<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    batch_labels: &[usize],
    binning: &str,
    n_bins: usize,
    verbose: usize,
) -> Result<Vec<HvgDispersionRes>, BixverseErrors> {
    get_hvg_dispersion_batch_aware(reader, cell_indices, batch_labels, binning, n_bins, verbose)
}

/// MVB HVG, batch-aware (streaming)
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep
/// * `batch_labels` - Batch assignment for each cell (same length as cell_
///   indices)
/// * `binning` - The binning strategy to use. One of
///   `"equal_width"` or `"equal_freq"`
/// * `n_bins` - Number of bins
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A `Vec<HvgDispersionRes>` with each element being a batch.
pub fn get_hvg_mvb_batch_aware_streaming<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    batch_labels: &[usize],
    binning: &str,
    n_bins: usize,
    verbose: usize,
) -> Result<Vec<HvgDispersionRes>, BixverseErrors> {
    get_hvg_dispersion_batch_aware_streaming(
        reader,
        cell_indices,
        batch_labels,
        binning,
        n_bins,
        verbose,
    )
}
