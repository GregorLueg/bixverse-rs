//! Implementations of highly variable gene detections in single cell.

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::time::Instant;
use thousands::Separable;

use crate::core::base::loess::*;
use crate::prelude::*;

/////////
// HVG //
/////////

/// Structure that stores HVG information
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

/// Enum for the different methods
pub enum HvgMethod {
    /// Variance stabilising transformation
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

/// Calculate the mean and variance of a CSC gene chunk
///
/// ### Params
///
/// * `chunk` - The `CscGeneChunk` representing the gene for which to calculate
///   the mean and variance
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

    // Only process cells that are in the filter
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

/// Implementation of the variance stabilised version of the HVG selection
///
/// ### Params
///
/// * `f_path` - Path to the gene-based binary file
/// * `cell_indices` - HashSet with the cell indices to keep.
/// * `loess_span` - Span parameter for the loess function
/// * `clip_max` - Optional clip max parameter
/// * `verbose` - If verbose, returns the timings of the function.
///
/// ### Returns
///
/// The `HvgRes`
pub fn get_hvg_vst(
    f_path: &str,
    cell_indices: &[usize],
    loess_span: f32,
    clip_max: Option<f32>,
    verbose: bool,
) -> HvgRes {
    let start_total = Instant::now();

    // Get data
    let start_read = Instant::now();

    let reader = ParallelSparseReader::new(f_path).unwrap();
    let mut gene_chunks: Vec<CscGeneChunk> = reader.get_all_genes();
    let no_cells = cell_indices.len();

    // build cell mapping ONCE... Before I was doing stupid shit
    let cell_idx_map: FxHashMap<u32, u32> = cell_indices
        .iter()
        .enumerate()
        .map(|(new_idx, &old_idx)| (old_idx as u32, new_idx as u32))
        .collect();

    let end_read = start_read.elapsed();

    if verbose {
        println!("Load in data: {:.2?}", end_read);
    }

    let start_gene_stats = Instant::now();

    let results: Vec<(f32, f32)> = gene_chunks
        .par_iter_mut()
        .map(|chunk| calculate_mean_var_filtered(chunk, &cell_idx_map, no_cells))
        .collect();

    let end_gene_stats = start_gene_stats.elapsed();

    if verbose {
        println!("Calculated gene statistics: {:.2?}", end_gene_stats);
    }

    let start_loess = Instant::now();

    let (means, vars): (Vec<f32>, Vec<f32>) = results.into_iter().unzip();

    let clip_max = clip_max.unwrap_or((no_cells as f32).sqrt());
    let means_log10: Vec<f32> = means.iter().map(|x| x.log10()).collect();
    let vars_log10: Vec<f32> = vars.iter().map(|x| x.log10()).collect();

    let loess = LoessRegression::new(loess_span, 2);
    let loess_res = loess.fit(&means_log10, &vars_log10);

    let end_loess = start_loess.elapsed();

    if verbose {
        println!("Fitted Loess: {:.2?}", end_loess);
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

    if verbose {
        println!("Standardised variance: {:.2?}", end_standard);
    }

    let total = start_total.elapsed();

    if verbose {
        println!("Total run time HVG detection: {:.2?}", total);
    }

    // transform to f64 for R
    HvgRes {
        mean: means.r_float_convert(),
        var: vars.r_float_convert(),
        var_exp: loess_res.fitted_vals.r_float_convert(),
        var_std: var_standardised.r_float_convert(),
    }
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
/// * `f_path` - Path to the gene-based binary file
/// * `cell_indices` - Slice with the cell indices to keep.
/// * `loess_span` - Span parameter for the loess function
/// * `clip_max` - Optional clip max parameter
/// * `verbose` - If verbose, returns the timings of the function.
///
/// ### Returns
///
/// The `HvgRes`
pub fn get_hvg_vst_streaming(
    f_path: &str,
    cell_indices: &[usize],
    loess_span: f32,
    clip_max: Option<f32>,
    verbose: bool,
) -> HvgRes {
    let start_total = Instant::now();

    let reader = ParallelSparseReader::new(f_path).unwrap();
    let header = reader.get_header();
    let no_genes = header.total_genes;
    let no_cells = cell_indices.len();

    let cell_idx_map: FxHashMap<u32, u32> = cell_indices
        .iter()
        .enumerate()
        .map(|(new_idx, &old_idx)| (old_idx as u32, new_idx as u32))
        .collect();

    if verbose {
        println!(
            "Pass 1/2: Calculating mean and variance for {} genes...",
            no_genes.separate_with_underscores()
        );
    }

    // Pass 1: Calculate mean and variance in batches
    let start_pass1 = Instant::now();

    const GENE_BATCH_SIZE: usize = 1000;
    let num_batches = no_genes.div_ceil(GENE_BATCH_SIZE);

    let mut means = Vec::with_capacity(no_genes);
    let mut vars = Vec::with_capacity(no_genes);

    for batch_idx in 0..num_batches {
        if verbose && batch_idx % 5 == 0 {
            let progress = (batch_idx + 1) as f32 / num_batches as f32 * 100.0;
            println!("  Progress: {:.1}%", progress);
        }

        let start_gene = batch_idx * GENE_BATCH_SIZE;
        let end_gene = ((batch_idx + 1) * GENE_BATCH_SIZE).min(no_genes);
        let gene_indices: Vec<usize> = (start_gene..end_gene).collect();

        let start_loading = Instant::now();

        let mut genes = reader.read_gene_parallel(&gene_indices);

        let end_loading = start_loading.elapsed();

        if verbose {
            println!("   Loaded batch in: {:.2?}.", end_loading);
        }

        let start_batch = Instant::now();

        let batch_results: Vec<(f32, f32)> = genes
            .par_iter_mut()
            .map(|gene| calculate_mean_var_filtered(gene, &cell_idx_map, no_cells))
            .collect();

        let end_batch = start_batch.elapsed();

        if verbose {
            println!("   Finished calculations in: {:.2?}.", end_batch);
        }

        for (mean, var) in batch_results {
            means.push(mean);
            vars.push(var);
        }
        // genes vec dropped here - memory freed
    }

    let end_pass1 = start_pass1.elapsed();

    if verbose {
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

    if verbose {
        println!("  Fitted Loess: {:.2?}", end_loess);
        println!("Pass 2/2: Calculating standardised variance...");
    }

    // Pass 2: Calculate standardised variance in batches
    let start_pass2 = Instant::now();

    let mut var_standardised = Vec::with_capacity(no_genes);

    for batch_idx in 0..num_batches {
        if verbose && batch_idx % 5 == 0 {
            let progress = (batch_idx + 1) as f32 / num_batches as f32 * 100.0;
            println!("  Progress: {:.1}%", progress);
        }

        let start_gene = batch_idx * GENE_BATCH_SIZE;
        let end_gene = ((batch_idx + 1) * GENE_BATCH_SIZE).min(no_genes);
        let gene_indices: Vec<usize> = (start_gene..end_gene).collect();

        let mut genes = reader.read_gene_parallel(&gene_indices);

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

        if verbose {
            println!(
                "   Finished calculating standardised variance in: {:.2?}.",
                end_batch
            );
        }

        var_standardised.extend(batch_std_vars);
    }

    let end_pass2 = start_pass2.elapsed();

    if verbose {
        println!(
            "  Calculated standardised variance total: {:.2?}",
            end_pass2
        );
    }

    let total = start_total.elapsed();

    if verbose {
        println!("Total run time HVG detection: {:.2?}", total);
    }

    // transform to f64 for R
    HvgRes {
        mean: means.r_float_convert(),
        var: vars.r_float_convert(),
        var_exp: loess_res.fitted_vals.r_float_convert(),
        var_std: var_standardised.r_float_convert(),
    }
}

/// To be implemented
pub fn get_hvg_dispersion() -> HvgRes {
    todo!("Dispersion method not yet implemented");

    #[allow(unreachable_code)]
    HvgRes {
        mean: Vec::new(),
        var: Vec::new(),
        var_exp: Vec::new(),
        var_std: Vec::new(),
    }
}

/// To be implemented
pub fn get_hvg_dispersion_streaming() -> HvgRes {
    todo!("Dispersion method with streaming not yet implemented");

    #[allow(unreachable_code)]
    HvgRes {
        mean: Vec::new(),
        var: Vec::new(),
        var_exp: Vec::new(),
        var_std: Vec::new(),
    }
}

/// To be implemented
pub fn get_hvg_mvb() -> HvgRes {
    todo!("MeanVarianceBin method not yet implemented");

    #[allow(unreachable_code)]
    HvgRes {
        mean: Vec::new(),
        var: Vec::new(),
        var_exp: Vec::new(),
        var_std: Vec::new(),
    }
}

/// To be implemented
pub fn get_hvg_mvb_streaming() -> HvgRes {
    todo!("MeanVarianceBin method with streaming not yet implemented");

    #[allow(unreachable_code)]
    HvgRes {
        mean: Vec::new(),
        var: Vec::new(),
        var_exp: Vec::new(),
        var_std: Vec::new(),
    }
}

/////////////////////
// HVG batch aware //
/////////////////////

/// Batch-aware HVG selection using VST method
///
/// Calculates HVG statistics separately for each batch, returning per-batch results.
///
/// ### Params
///
/// * `f_path` - Path to the gene-based binary file
/// * `cell_indices` - Slice with the cell indices to keep
/// * `batch_labels` - Batch assignment for each cell (same length as cell_indices)
/// * `loess_span` - Span parameter for the loess function
/// * `clip_max` - Optional clip max parameter
/// * `verbose` - If verbose, prints timing information
///
/// ### Returns
///
/// `Vec<HvgRes>` - One HvgRes per batch
pub fn get_hvg_vst_batch_aware(
    f_path: &str,
    cell_indices: &[usize],
    batch_labels: &[usize],
    loess_span: f32,
    clip_max: Option<f32>,
    verbose: bool,
) -> Vec<HvgRes> {
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

    if verbose {
        println!("Setup batch maps: {:.2?}", end_setup);
        println!("Processing {} batches", n_batches);
    }

    // load data
    let start_read = Instant::now();
    let reader = ParallelSparseReader::new(f_path).unwrap();
    let mut gene_chunks: Vec<CscGeneChunk> = reader.get_all_genes();
    let end_read = start_read.elapsed();

    if verbose {
        println!("Loaded data: {:.2?}", end_read);
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

    if verbose {
        println!("Calculated gene statistics per batch: {:.2?}", end_pass_1);
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

    if verbose {
        println!("Fitted loess per batch: {:.2?}", end_loess);
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

    if verbose {
        println!(
            "Calculated standardised variance per batch: {:.2?}",
            end_pass_2
        );
    }

    let total = start_total.elapsed();

    if verbose {
        println!("Total runtime batch-aware HVG: {:.2?}", total);
    }

    batch_means
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
        .collect()
}

/// Batch-aware HVG selection using VST method with streaming
///
/// Two-pass approach processing genes in batches to minimise memory usage.
/// Calculates HVG statistics separately for each batch.
///
/// ### Params
///
/// * `f_path` - Path to the gene-based binary file
/// * `cell_indices` - Slice with the cell indices to keep
/// * `batch_labels` - Batch assignment for each cell (same length as cell_indices)
/// * `loess_span` - Span parameter for the loess function
/// * `clip_max` - Optional clip max parameter
/// * `verbose` - If verbose, prints timing information
///
/// ### Returns
///
/// `Vec<HvgRes>` - One HvgRes per batch
pub fn get_hvg_vst_batch_aware_streaming(
    f_path: &str,
    cell_indices: &[usize],
    batch_labels: &[usize],
    loess_span: f32,
    clip_max: Option<f32>,
    verbose: bool,
) -> Vec<HvgRes> {
    let start_total = Instant::now();

    let reader = ParallelSparseReader::new(f_path).unwrap();
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

    if verbose {
        println!("Setup batch maps: {:.2?}", end_setup);
        println!("Processing {} batches", n_batches);
    }

    // Pass 1: Calculate mean and variance per batch
    if verbose {
        println!(
            "Pass 1/2: Calculating mean and variance for {} genes...",
            no_genes.separate_with_underscores()
        );
    }

    let start_pass1 = Instant::now();
    const GENE_BATCH_SIZE: usize = 1000;
    let num_batches = no_genes.div_ceil(GENE_BATCH_SIZE);

    let mut batch_means: Vec<Vec<f32>> = vec![Vec::with_capacity(no_genes); n_batches];
    let mut batch_vars: Vec<Vec<f32>> = vec![Vec::with_capacity(no_genes); n_batches];

    for chunk_idx in 0..num_batches {
        if verbose && chunk_idx % 5 == 0 {
            let progress = (chunk_idx + 1) as f32 / num_batches as f32 * 100.0;
            println!("  Progress: {:.1}%", progress);
        }

        let start_gene = chunk_idx * GENE_BATCH_SIZE;
        let end_gene = ((chunk_idx + 1) * GENE_BATCH_SIZE).min(no_genes);
        let gene_indices: Vec<usize> = (start_gene..end_gene).collect();

        let mut genes = reader.read_gene_parallel(&gene_indices);

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

    if verbose {
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

    if verbose {
        println!("  Fitted loess per batch: {:.2?}", end_loess);
        println!("Pass 2/2: Calculating standardised variance...");
    }

    // Pass 2: Calculate standardised variance per batch
    let start_pass2 = Instant::now();

    let mut batch_std_vars: Vec<Vec<f32>> = vec![Vec::with_capacity(no_genes); n_batches];

    for chunk_idx in 0..num_batches {
        if verbose && chunk_idx % 5 == 0 {
            let progress = (chunk_idx + 1) as f32 / num_batches as f32 * 100.0;
            println!("  Progress: {:.1}%", progress);
        }

        let start_gene = chunk_idx * GENE_BATCH_SIZE;
        let end_gene = ((chunk_idx + 1) * GENE_BATCH_SIZE).min(no_genes);
        let gene_indices: Vec<usize> = (start_gene..end_gene).collect();

        let mut genes = reader.read_gene_parallel(&gene_indices);

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

    if verbose {
        println!(
            "  Calculated standardised variance per batch: {:.2?}",
            end_pass2
        );
    }

    let total = start_total.elapsed();

    if verbose {
        println!("Total runtime batch-aware HVG: {:.2?}", total);
    }

    // Build results per batch
    batch_means
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
        .collect()
}
