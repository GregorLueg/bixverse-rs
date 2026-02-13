use indexmap::IndexSet;
use rayon::prelude::*;
use std::time::Instant;
use thousands::Separable;

use crate::core::base::loess::*;
use crate::prelude::*;

/////////
// HVG //
/////////

/////////////
// Helpers //
/////////////

/// Structure that stores HVG information
///
/// ### Fields
///
/// * `mean` - Mean expression of the gene.
/// * `var` - Detected variance of the gene.
/// * `var_exp` - Expected variance of the gene.
/// * `var_std` - Standardised variance of the gene.
#[derive(Clone, Debug)]
pub struct HvgRes {
    pub mean: Vec<f64>,
    pub var: Vec<f64>,
    pub var_exp: Vec<f64>,
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

/// Calculate the mean and variance of a pre-filtered CSC gene chunk.
///
/// Assumes `filter_selected_cells` has already been called on the chunk,
/// so all entries are relevant and no lookup is needed.
///
/// ### Params
///
/// * `gene` - The `CscGeneChunk` representing the gene for which to calculate
///   the mean and variance.
/// * `no_cells` - The number of cells in the chunk
///
/// ### Returns
///
/// Tuple of `(mean, var)`
#[inline]
pub fn calculate_mean_var(gene: &CscGeneChunk, no_cells: usize) -> (f32, f32) {
    let no_cells_f = no_cells as f32;
    let nnz = gene.indices.len();

    let mut sum = 0f32;
    for i in 0..nnz {
        sum += gene.data_raw[i].to_f32();
    }

    let mean = sum / no_cells_f;

    let mut sum_sq_diff = 0f32;
    for i in 0..nnz {
        let diff = gene.data_raw[i].to_f32() - mean;
        sum_sq_diff += diff * diff;
    }

    let n_zero = no_cells_f - nnz as f32;
    let var = (sum_sq_diff + n_zero * mean * mean) / no_cells_f;

    (mean, var)
}

/// Calculate the standardised variance of a pre-filtered CSC gene chunk.
///
/// Assumes `filter_selected_cells` has already been called on the chunk.
///
/// ### Params
///
/// * `gene`: The pre-filtered CSC gene chunk.
/// * `mean`: The mean of the gene chunk.
/// * `expected_var`: The expected variance of the gene chunk.
/// * `clip_max`: The maximum value to clip the standardised variance to.
/// * `no_cells`: The number of cells in the gene chunk.
///
/// ### Returns
///
/// The standardised variance of the gene chunk.
#[inline]
pub fn calculate_std_variance(
    gene: &CscGeneChunk,
    mean: f32,
    expected_var: f32,
    clip_max: f32,
    no_cells: usize,
) -> f32 {
    let no_cells_f = no_cells as f32;
    let expected_sd = expected_var.sqrt();
    let nnz = gene.indices.len();

    let mut sum_std = 0f32;
    let mut sum_sq_std = 0f32;

    for i in 0..nnz {
        let val = gene.data_raw[i].to_f32();
        let norm = ((val - mean) / expected_sd).clamp(-clip_max, clip_max);
        sum_std += norm;
        sum_sq_std += norm * norm;
    }

    let n_zeros = no_cells - nnz;
    if n_zeros > 0 {
        let std_zero = ((-mean) / expected_sd).clamp(-clip_max, clip_max);
        sum_std += n_zeros as f32 * std_zero;
        sum_sq_std += n_zeros as f32 * std_zero * std_zero;
    }

    let std_mean = sum_std / no_cells_f;
    (sum_sq_std / no_cells_f) - (std_mean * std_mean)
}

/// Calculate mean and var per batch in a single pass over a filtered gene chunk.
///
/// Uses one-pass sum/sum_sq accumulation per batch to avoid N_batches separate
/// passes.
///
/// ### Params
///
/// * `gene`: The gene chunk to calculate mean and var for.
/// * `cell_batch_labels`: The batch labels for each cell.
/// * `batch_sizes`: The size of each batch.
/// * `n_batches`: The number of batches.
///
/// ### Returns
///
/// A vector of `(mean, var)` tuples for each batch.
#[inline]
pub fn calculate_mean_var_batched(
    gene: &CscGeneChunk,
    cell_batch_labels: &[usize],
    batch_sizes: &[usize],
    n_batches: usize,
) -> Vec<(f32, f32)> {
    let mut batch_sum = vec![0f32; n_batches];
    let mut batch_sum_sq = vec![0f32; n_batches];
    let mut batch_nnz = vec![0usize; n_batches];

    for i in 0..gene.indices.len() {
        let b = cell_batch_labels[gene.indices[i] as usize];
        let val = gene.data_raw[i].to_f32();
        batch_sum[b] += val;
        batch_sum_sq[b] += val * val;
        batch_nnz[b] += 1;
    }

    (0..n_batches)
        .map(|b| {
            let n = batch_sizes[b] as f32;
            let mean = batch_sum[b] / n;
            let var = batch_sum_sq[b] / n - mean * mean;
            (mean, var)
        })
        .collect()
}

/// Calculate standardised var per batch in a single pass over a gene chunk.
///
/// Assumes that the gene chunk was filtered.
///
/// ### Params
///
/// * `gene`: The gene chunk to process.
/// * `cell_batch_labels`: The batch labels for each cell.
/// * `batch_means`: The mean values for each batch.
/// * `batch_expected_sds`: The expected standard deviations for each batch.
/// * `batch_clip_maxs`: The maximum values for clipping for each batch.
/// * `batch_sizes`: The sizes of each batch.
/// * `n_batches`: The number of batches.
#[inline]
pub fn calculate_std_variance_batched(
    gene: &CscGeneChunk,
    cell_batch_labels: &[usize],
    batch_means: &[f32],
    batch_expected_sds: &[f32],
    batch_clip_maxs: &[f32],
    batch_sizes: &[usize],
    n_batches: usize,
) -> Vec<f32> {
    let mut batch_sum_std = vec![0f32; n_batches];
    let mut batch_sum_sq_std = vec![0f32; n_batches];
    let mut batch_nnz = vec![0usize; n_batches];

    for i in 0..gene.indices.len() {
        let b = cell_batch_labels[gene.indices[i] as usize];
        let val = gene.data_raw[i].to_f32();
        let norm = ((val - batch_means[b]) / batch_expected_sds[b])
            .clamp(-batch_clip_maxs[b], batch_clip_maxs[b]);
        batch_sum_std[b] += norm;
        batch_sum_sq_std[b] += norm * norm;
        batch_nnz[b] += 1;
    }

    (0..n_batches)
        .map(|b| {
            let n = batch_sizes[b] as f32;
            let n_zeros = batch_sizes[b] - batch_nnz[b];

            let mut sum_std = batch_sum_std[b];
            let mut sum_sq_std = batch_sum_sq_std[b];

            if n_zeros > 0 {
                let std_zero = ((-batch_means[b]) / batch_expected_sds[b])
                    .clamp(-batch_clip_maxs[b], batch_clip_maxs[b]);
                sum_std += n_zeros as f32 * std_zero;
                sum_sq_std += n_zeros as f32 * std_zero * std_zero;
            }

            let std_mean = sum_std / n;
            (sum_sq_std / n) - (std_mean * std_mean)
        })
        .collect()
}

//////////////////////
// HVG single batch //
//////////////////////

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

    let start_read = Instant::now();
    let reader = ParallelSparseReader::new(f_path).unwrap();
    let mut gene_chunks: Vec<CscGeneChunk> = reader.get_all_genes();
    let no_cells = cell_indices.len();
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();
    let end_read = start_read.elapsed();

    if verbose {
        println!("Load in data: {:.2?}", end_read);
    }

    let start_filter = Instant::now();
    gene_chunks
        .par_iter_mut()
        .for_each(|chunk| chunk.filter_selected_cells(&cell_set));
    let end_filter = start_filter.elapsed();

    if verbose {
        println!("Filtered cells: {:.2?}", end_filter);
    }

    let start_gene_stats = Instant::now();
    let results: Vec<(f32, f32)> = gene_chunks
        .par_iter()
        .map(|chunk| calculate_mean_var(chunk, no_cells))
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
        .map(|((chunk, var_fitted), mean)| {
            let expected_var = 10_f32.powf(*var_fitted);
            calculate_std_variance(chunk, *mean, expected_var, clip_max, no_cells)
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
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();

    if verbose {
        println!(
            "Pass 1/2: Calculating mean and variance for {} genes...",
            no_genes.separate_with_underscores()
        );
    }

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

        let mut genes = reader.read_gene_parallel(&gene_indices);

        genes
            .par_iter_mut()
            .for_each(|chunk| chunk.filter_selected_cells(&cell_set));

        let batch_results: Vec<(f32, f32)> = genes
            .par_iter()
            .map(|gene| calculate_mean_var(gene, no_cells))
            .collect();

        for (mean, var) in batch_results {
            means.push(mean);
            vars.push(var);
        }
    }

    let end_pass1 = start_pass1.elapsed();

    if verbose {
        println!("  Calculated gene statistics: {:.2?}", end_pass1);
    }

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

        genes
            .par_iter_mut()
            .for_each(|chunk| chunk.filter_selected_cells(&cell_set));

        let batch_std_vars: Vec<f32> = genes
            .par_iter()
            .enumerate()
            .map(|(local_idx, gene)| {
                let gene_idx = start_gene + local_idx;
                let expected_var = 10_f32.powf(loess_res.fitted_vals[gene_idx]);
                calculate_std_variance(gene, means[gene_idx], expected_var, clip_max, no_cells)
            })
            .collect();

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

    let start_setup = Instant::now();
    let n_batches = *batch_labels.iter().max().unwrap() + 1;
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();

    // cell_batch_labels[new_idx] = batch for that cell
    // IndexSet preserves insertion order, so new_idx matches cell_indices order
    let cell_batch_labels: Vec<usize> = batch_labels.to_vec();

    let mut batch_sizes = vec![0usize; n_batches];
    for &b in batch_labels {
        batch_sizes[b] += 1;
    }

    let end_setup = start_setup.elapsed();

    if verbose {
        println!("Setup: {:.2?}", end_setup);
        println!("Processing {} batches", n_batches);
    }

    let start_read = Instant::now();
    let reader = ParallelSparseReader::new(f_path).unwrap();
    let mut gene_chunks: Vec<CscGeneChunk> = reader.get_all_genes();
    let end_read = start_read.elapsed();

    if verbose {
        println!("Loaded data: {:.2?}", end_read);
    }

    let start_filter = Instant::now();
    gene_chunks
        .par_iter_mut()
        .for_each(|chunk| chunk.filter_selected_cells(&cell_set));
    let end_filter = start_filter.elapsed();

    if verbose {
        println!("Filtered cells: {:.2?}", end_filter);
    }

    // Pass 1: mean/var for all batches in a single pass per gene
    let start_pass1 = Instant::now();
    let all_mean_var: Vec<Vec<(f32, f32)>> = gene_chunks
        .par_iter()
        .map(|chunk| calculate_mean_var_batched(chunk, &cell_batch_labels, &batch_sizes, n_batches))
        .collect();

    // Transpose: gene-major -> batch-major
    let mut batch_means: Vec<Vec<f32>> = vec![Vec::with_capacity(gene_chunks.len()); n_batches];
    let mut batch_vars: Vec<Vec<f32>> = vec![Vec::with_capacity(gene_chunks.len()); n_batches];

    for gene_results in &all_mean_var {
        for (b, &(mean, var)) in gene_results.iter().enumerate() {
            batch_means[b].push(mean);
            batch_vars[b].push(var);
        }
    }

    let end_pass1 = start_pass1.elapsed();

    if verbose {
        println!("Calculated gene statistics per batch: {:.2?}", end_pass1);
    }

    // Fit loess per batch
    let start_loess = Instant::now();
    let mut batch_loess_results = Vec::with_capacity(n_batches);
    let mut batch_clip_maxs = Vec::with_capacity(n_batches);

    for b in 0..n_batches {
        let clip = clip_max.unwrap_or((batch_sizes[b] as f32).sqrt());
        batch_clip_maxs.push(clip);

        let means_log10: Vec<f32> = batch_means[b].iter().map(|x| x.log10()).collect();
        let vars_log10: Vec<f32> = batch_vars[b].iter().map(|x| x.log10()).collect();

        let loess = LoessRegression::new(loess_span, 2);
        let loess_res = loess.fit(&means_log10, &vars_log10);
        batch_loess_results.push(loess_res);
    }

    let end_loess = start_loess.elapsed();

    if verbose {
        println!("Fitted loess per batch: {:.2?}", end_loess);
    }

    // Pass 2: standardised variance for all batches in a single pass per gene
    let start_pass2 = Instant::now();

    // Pre-compute expected SDs per batch per gene
    let batch_expected_sds: Vec<Vec<f32>> = (0..n_batches)
        .map(|b| {
            batch_loess_results[b]
                .fitted_vals
                .iter()
                .map(|v| 10_f32.powf(*v).sqrt())
                .collect()
        })
        .collect();

    let all_std_vars: Vec<Vec<f32>> = gene_chunks
        .par_iter()
        .enumerate()
        .map(|(gene_idx, chunk)| {
            let gene_means: Vec<f32> = (0..n_batches).map(|b| batch_means[b][gene_idx]).collect();
            let gene_sds: Vec<f32> = (0..n_batches)
                .map(|b| batch_expected_sds[b][gene_idx])
                .collect();

            calculate_std_variance_batched(
                chunk,
                &cell_batch_labels,
                &gene_means,
                &gene_sds,
                &batch_clip_maxs,
                &batch_sizes,
                n_batches,
            )
        })
        .collect();

    // Transpose: gene-major -> batch-major
    let mut batch_std_vars: Vec<Vec<f32>> = vec![Vec::with_capacity(gene_chunks.len()); n_batches];
    for gene_results in &all_std_vars {
        for (b, &sv) in gene_results.iter().enumerate() {
            batch_std_vars[b].push(sv);
        }
    }

    let end_pass2 = start_pass2.elapsed();

    if verbose {
        println!(
            "Calculated standardised variance per batch: {:.2?}",
            end_pass2
        );
    }

    let total = start_total.elapsed();

    if verbose {
        println!("Total runtime batch-aware HVG: {:.2?}", total);
    }

    (0..n_batches)
        .map(|b| HvgRes {
            mean: batch_means[b].clone().r_float_convert(),
            var: batch_vars[b].clone().r_float_convert(),
            var_exp: batch_loess_results[b].fitted_vals.clone().r_float_convert(),
            var_std: batch_std_vars[b].clone().r_float_convert(),
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

    let start_setup = Instant::now();
    let n_batches = *batch_labels.iter().max().unwrap() + 1;
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();
    let cell_batch_labels: Vec<usize> = batch_labels.to_vec();

    let mut batch_sizes = vec![0usize; n_batches];
    for &b in batch_labels {
        batch_sizes[b] += 1;
    }

    let end_setup = start_setup.elapsed();

    if verbose {
        println!("Setup: {:.2?}", end_setup);
        println!("Processing {} batches", n_batches);
        println!(
            "Pass 1/2: Calculating mean and variance for {} genes...",
            no_genes.separate_with_underscores()
        );
    }

    // Pass 1: mean/var
    let start_pass1 = Instant::now();
    const GENE_BATCH_SIZE: usize = 1000;
    let num_gene_batches = no_genes.div_ceil(GENE_BATCH_SIZE);

    let mut batch_means: Vec<Vec<f32>> = vec![Vec::with_capacity(no_genes); n_batches];
    let mut batch_vars: Vec<Vec<f32>> = vec![Vec::with_capacity(no_genes); n_batches];

    for chunk_idx in 0..num_gene_batches {
        if verbose && chunk_idx % 5 == 0 {
            let progress = (chunk_idx + 1) as f32 / num_gene_batches as f32 * 100.0;
            println!("  Progress: {:.1}%", progress);
        }

        let start_gene = chunk_idx * GENE_BATCH_SIZE;
        let end_gene = ((chunk_idx + 1) * GENE_BATCH_SIZE).min(no_genes);
        let gene_indices: Vec<usize> = (start_gene..end_gene).collect();

        let mut genes = reader.read_gene_parallel(&gene_indices);

        genes
            .par_iter_mut()
            .for_each(|chunk| chunk.filter_selected_cells(&cell_set));

        let chunk_results: Vec<Vec<(f32, f32)>> = genes
            .par_iter()
            .map(|gene| {
                calculate_mean_var_batched(gene, &cell_batch_labels, &batch_sizes, n_batches)
            })
            .collect();

        for gene_results in &chunk_results {
            for (b, &(mean, var)) in gene_results.iter().enumerate() {
                batch_means[b].push(mean);
                batch_vars[b].push(var);
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
    let mut batch_clip_maxs = Vec::with_capacity(n_batches);

    for b in 0..n_batches {
        let clip = clip_max.unwrap_or((batch_sizes[b] as f32).sqrt());
        batch_clip_maxs.push(clip);

        let means_log10: Vec<f32> = batch_means[b].iter().map(|x| x.log10()).collect();
        let vars_log10: Vec<f32> = batch_vars[b].iter().map(|x| x.log10()).collect();

        let loess = LoessRegression::new(loess_span, 2);
        let loess_res = loess.fit(&means_log10, &vars_log10);
        batch_loess_results.push(loess_res);
    }

    let end_loess = start_loess.elapsed();

    if verbose {
        println!("  Fitted loess per batch: {:.2?}", end_loess);
        println!("Pass 2/2: Calculating standardised variance...");
    }

    // Pre-compute expected SDs per batch per gene
    let batch_expected_sds: Vec<Vec<f32>> = (0..n_batches)
        .map(|b| {
            batch_loess_results[b]
                .fitted_vals
                .iter()
                .map(|v| 10_f32.powf(*v).sqrt())
                .collect()
        })
        .collect();

    // Pass 2: standardised variance
    let start_pass2 = Instant::now();
    let mut batch_std_vars: Vec<Vec<f32>> = vec![Vec::with_capacity(no_genes); n_batches];

    for chunk_idx in 0..num_gene_batches {
        if verbose && chunk_idx % 5 == 0 {
            let progress = (chunk_idx + 1) as f32 / num_gene_batches as f32 * 100.0;
            println!("  Progress: {:.1}%", progress);
        }

        let start_gene = chunk_idx * GENE_BATCH_SIZE;
        let end_gene = ((chunk_idx + 1) * GENE_BATCH_SIZE).min(no_genes);
        let gene_indices: Vec<usize> = (start_gene..end_gene).collect();

        let mut genes = reader.read_gene_parallel(&gene_indices);

        genes
            .par_iter_mut()
            .for_each(|chunk| chunk.filter_selected_cells(&cell_set));

        let chunk_std_vars: Vec<Vec<f32>> = genes
            .par_iter()
            .enumerate()
            .map(|(local_idx, gene)| {
                let gene_idx = start_gene + local_idx;
                let gene_means: Vec<f32> =
                    (0..n_batches).map(|b| batch_means[b][gene_idx]).collect();
                let gene_sds: Vec<f32> = (0..n_batches)
                    .map(|b| batch_expected_sds[b][gene_idx])
                    .collect();

                calculate_std_variance_batched(
                    gene,
                    &cell_batch_labels,
                    &gene_means,
                    &gene_sds,
                    &batch_clip_maxs,
                    &batch_sizes,
                    n_batches,
                )
            })
            .collect();

        for gene_results in &chunk_std_vars {
            for (b, &sv) in gene_results.iter().enumerate() {
                batch_std_vars[b].push(sv);
            }
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

    (0..n_batches)
        .map(|b| HvgRes {
            mean: batch_means[b].clone().r_float_convert(),
            var: batch_vars[b].clone().r_float_convert(),
            var_exp: batch_loess_results[b].fitted_vals.clone().r_float_convert(),
            var_std: batch_std_vars[b].clone().r_float_convert(),
        })
        .collect()
}
