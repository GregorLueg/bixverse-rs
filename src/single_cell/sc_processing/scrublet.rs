//! Contains the Scrublet method for detections of doublets in single cell,
//! see Wollock, et al., Cell Syst., 2019

use faer::{Mat, MatRef, concat};
use half::f16;
use indexmap::IndexSet;
use rand::prelude::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

use crate::core::math::pca_svd::*;
use crate::core::math::sparse::sparse_svd_lanczos;
use crate::prelude::*;
use crate::single_cell::sc_processing::hvg::*;
use crate::single_cell::sc_processing::pca::*;

///////////
// Types //
///////////

/// Type alias for Scrublet PCA results
///
/// ### Fields
///
/// * `0` - PCA scores
/// * `1` - PCA loadings
/// * `2` - Gene means
/// * `3` - Gene standard deviations
type ScrubletPcaRes = (Mat<f32>, Mat<f32>, Vec<f32>, Vec<f32>);

/// Type alias for Scrublet Doublet Scores
///
/// ### Fields
///
/// * `0` - Scores actual cells
/// * `1` - Errors actual cells
/// * `2` - Scores simulated cells
/// * `3` - Errors simulated cells
type ScrubletDoubletScores = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// Type alias for final Scrublet results
///
/// ### Fields
///
/// * `0` - The actual Scrublet results from the algorithm
/// * `1` - Optional PCA embeddings across observed and simulated cells
/// * `2` - First parent of the simulated parent
/// * `3` - Second parent of the simualted parent
pub type FinalScrubletRes = (
    ScrubletResult,
    Option<Mat<f32>>,
    Option<Vec<usize>>,
    Option<Vec<usize>>,
);

////////////////////////
// Params and results //
////////////////////////

/// Structure to store the Scrublet parameters
///
/// **Doublet Simulation:**
///
/// * `sim_doublet_ratio` - Number of doublets to simulate relative to the
///   number of observed cells (e.g., 2.0 simulates 2x as many doublets).
/// * `expected_doublet_rate` - Expected doublet rate for the experiment
///   (typically 0.05-0.10 depending on cell loading).
/// * `stdev_doublet_rate` - Uncertainty in the expected doublet rate.
///
/// **Doublet calling:**
///
/// * `n_bins` - Number of bins for histogram-based automatic threshold
///   detection (typically 50-100).
/// * `manual_threshold` - Optional manual doublet score threshold. If `None`,
///   threshold is automatically detected from simulated doublet score
///   distribution.
///
/// **PCA:**
///
/// * `no_pcs` - Number of principal components to use for embedding.
/// * `random_svd` - Whether to use randomized SVD (faster) vs exact SVD.
///
/// **kNN Graph:**
///
/// * `knn_params` - All the parameters related to the kNN graph generation
#[derive(Clone, Debug)]
pub struct ScrubletParams {
    /// Shall the counts be log-transformed
    pub log_transform: bool,
    /// Shall the data be mean-centred
    pub mean_center: bool,
    /// Shall the data be variance normalised
    pub normalise_variance: bool,
    /// Optional target size. If not provided, will default to the mean library
    /// size of the cells.
    pub target_size: Option<f32>,
    /// Percentile threshold for highly variable genes.
    pub min_gene_var_pctl: f32,
    /// Method for HVG selection. One of `"vst"`, `"mvb"`, or `"dispersion"`.
    pub hvg_method: String,
    /// Span parameter for loess fitting in VST method.
    pub loess_span: f64,
    /// Optional maximum value for clipping in variance stabilisation.
    pub clip_max: Option<f32>,
    /// Number of doublets to simulate relative to the number of observed cells
    /// (e.g., 2.0 simulates 2x as many doublets).
    pub sim_doublet_ratio: f32,
    /// Expected doublet rate for the experiment (typically 0.05-0.10 depending
    /// on cell loading).
    pub expected_doublet_rate: f32,
    /// Uncertainty in the expected doublet rate.
    pub stdev_doublet_rate: f32,
    /// Number of bins for histogram-based automatic threshold detection
    /// (typically 50-100).
    pub n_bins: usize,
    /// Optional manual doublet score threshold. If `None`, threshold is
    /// automatically detected from simulated doublet score distribution.
    pub manual_threshold: Option<f32>,
    /// Number of principal components to use for embedding.
    pub no_pcs: usize,
    /// Whether to use randomized SVD (faster) vs exact SVD.
    pub random_svd: bool,
    /// Parameters for the various approximate nearest neighbour searches
    /// in ann-search-rs
    pub knn_params: KnnParams,
}

/// Result structure for Scrublet doublet detection
///
/// Contains predictions, scores, and statistics from the Scrublet algorithm.
#[derive(Clone, Debug)]
pub struct ScrubletResult {
    /// Boolean vector indicating which observed cells are predicted as doublets
    /// (true = doublet, false = singlet).
    pub predicted_doublets: Vec<bool>,
    /// Doublet scores for each observed cell. Higher scores indicate higher
    /// likelihood of being a doublet.
    pub doublet_scores_obs: Vec<f32>,
    /// Doublet scores for simulated doublets. Used to determine the threshold
    /// and validate detection performance.
    pub doublet_scores_sim: Vec<f32>,
    /// Standard errors for doublet scores of observed cells. Indicates
    /// uncertainty in each score.
    pub doublet_errors_obs: Vec<f32>,
    /// Z-scores for observed cells, calculated as
    /// `(score - threshold) / error`. Higher absolute values indicate more
    /// confident predictions.
    pub z_scores: Vec<f32>,
    /// Doublet score threshold used to classify cells. Cells with scores above
    /// this value are called doublets.
    pub threshold: f32,
    /// Fraction of observed cells called as doublets.
    pub detected_doublet_rate: f32,
    /// Fraction of simulated doublets with scores above the threshold.
    /// Indicates what proportion of doublets can be detected.
    pub detectable_doublet_fraction: f32,
    /// Estimated overall doublet rate, calculated as
    /// `detected_doublet_rate / detectable_doublet_fraction`. Should roughly
    /// match the expected doublet rate if detection is working well.
    pub overall_doublet_rate: f32,
}

/////////////
// Helpers //
/////////////

impl CsrCellChunk {
    /// Add two cells together for Scrublet doublet simulation
    ///
    /// This combines two cells by:
    /// 1. Filtering both to HVG genes only
    /// 2. Adding their raw counts
    /// 3. Using the combined library size (from ALL genes) for normalisation
    /// 4. Normalising to target size
    ///
    /// ### Params
    ///
    /// * `cell1` - First cell
    /// * `cell2` - Second cell
    /// * `hvg_indices` - HashSet of HVG gene indices to keep
    /// * `target_size` - Target normalization size (e.g., 1e6 for CPM)
    /// * `log_transform` - Shall the counts be log-transformed.
    /// * `doublet_index` - Index to assign to the new doublet
    ///
    /// ### Returns
    ///
    /// New `CsrCellChunk` representing the doublet
    #[allow(clippy::too_many_arguments)]
    pub fn add_cells_scrublet(
        cell1: &CsrCellChunk,
        cell2: &CsrCellChunk,
        hvg_indices: &FxHashSet<usize>,
        hvg_lib_size_combined: usize,
        target_size: f32,
        log_transform: bool,
        doublet_index: usize,
    ) -> Self {
        let mut gene_counts: FxHashMap<u32, u32> = FxHashMap::default();

        for i in 0..cell1.indices.len() {
            let gene_idx = cell1.indices[i] as usize;
            if hvg_indices.contains(&gene_idx) {
                *gene_counts.entry(cell1.indices[i]).or_insert(0) += cell1.data_raw.get(i);
            }
        }
        for i in 0..cell2.indices.len() {
            let gene_idx = cell2.indices[i] as usize;
            if hvg_indices.contains(&gene_idx) {
                *gene_counts.entry(cell2.indices[i]).or_insert(0) += cell2.data_raw.get(i);
            }
        }

        let mut gene_vec: Vec<(u32, u32)> = gene_counts.into_iter().collect();
        gene_vec.sort_unstable_by_key(|&(gene, _)| gene);

        let mut data_raw = Vec::with_capacity(gene_vec.len());
        let mut data_norm = Vec::with_capacity(gene_vec.len());
        let mut indices = Vec::with_capacity(gene_vec.len());

        let norm_factor = target_size / hvg_lib_size_combined as f32;

        for (gene, count) in gene_vec {
            let count_u16 = count.min(u16::MAX as u32) as u16;
            data_raw.push(count_u16);

            let normalised = if log_transform {
                (count as f32 * norm_factor).ln_1p()
            } else {
                count as f32 * norm_factor
            };
            let normalised_clamped = normalised.clamp(-65504.0, 65504.0);
            data_norm.push(F16::from(f16::from_f32(normalised_clamped)));
            indices.push(gene);
        }

        Self {
            data_raw: RawCounts::U16(data_raw),
            data_norm,
            library_size: hvg_lib_size_combined,
            indices,
            original_index: doublet_index,
            to_keep: true,
        }
    }
}

/// Scale `Vec<CsrCellChunk>` using pre-calculated gene means and stds
///
/// This ensures simulated doublets are scaled using the SAME statistics
/// as the observed cells (critical for proper PCA projection)
///
/// ### Params
///
/// * `chunks` - Vector of cell chunks (simulated doublets)
/// * `gene_means` - Mean for each gene (from observed data)
/// * `gene_stds` - Std dev for each gene (from observed data)
/// * `mean_center` - Center the data around the mean
/// * `normalise_variance` - Normalise the variance
/// * `n_genes` - Total number of genes
///
/// ### Returns
///
/// Dense matrix (cells x genes) with z-scored values
pub fn scale_cell_chunks_with_stats(
    chunks: &[CsrCellChunk],
    gene_means: &[f32],
    gene_stds: &[f32],
    mean_center: bool,
    normalise_variance: bool,
    n_genes: usize,
) -> Mat<f32> {
    let n_cells = chunks.len();
    let mut result = Mat::<f32>::zeros(n_cells, n_genes);

    for (cell_idx, chunk) in chunks.iter().enumerate() {
        match (mean_center, normalise_variance) {
            (false, false) => {
                for i in 0..chunk.indices.len() {
                    let gene = chunk.indices[i] as usize;
                    let val = chunk.data_norm[i].to_f32();
                    *result.get_mut(cell_idx, gene) = val;
                }
            }
            (true, false) => {
                for gene in 0..n_genes {
                    *result.get_mut(cell_idx, gene) = -gene_means[gene];
                }
                for i in 0..chunk.indices.len() {
                    let gene = chunk.indices[i] as usize;
                    let val = chunk.data_norm[i].to_f32();
                    *result.get_mut(cell_idx, gene) = val - gene_means[gene];
                }
            }
            (false, true) => {
                for i in 0..chunk.indices.len() {
                    let gene = chunk.indices[i] as usize;
                    let val = chunk.data_norm[i].to_f32();
                    *result.get_mut(cell_idx, gene) = val / gene_stds[gene];
                }
            }
            (true, true) => {
                for gene in 0..n_genes {
                    *result.get_mut(cell_idx, gene) = -gene_means[gene] / gene_stds[gene];
                }
                for i in 0..chunk.indices.len() {
                    let gene = chunk.indices[i] as usize;
                    let val = chunk.data_norm[i].to_f32();
                    *result.get_mut(cell_idx, gene) = (val - gene_means[gene]) / gene_stds[gene];
                }
            }
        }
    }

    result
}

/// Calculate PCA for doublet detection methods using sparse SVD
///
/// Computes PCA on observed cells without densifying the full matrix. Gene
/// chunks are re-normalised using HVG-specific library sizes (matching the
/// Scrublet normalisation scheme), assembled into a CSC sparse matrix, and
/// decomposed via sparse randomised SVD or Lanczos. This avoids holding
/// an `n_cells x n_genes` dense matrix in memory, which is the primary
/// memory bottleneck in both Scrublet and Boost.
///
/// The returned gene means and standard deviations are computed from the
/// re-normalised sparse data and must be used to scale simulated doublets
/// before projecting them into the same PC space (via
/// `scale_cell_chunks_with_stats`).
///
/// ### Params
///
/// * `f_path_gene` - Path to the gene-based binary file (CSC on disk).
/// * `cell_indices` - Slice of cell indices to include in the analysis.
/// * `gene_indices` - Slice of gene indices (HVGs) to use.
/// * `hvg_library_sizes` - Per-cell library sizes computed over HVG genes
///   only. Must be in the same order as `cell_indices`.
/// * `target_size` - Normalisation target size (e.g. mean HVG library size).
/// * `log_transform` - Whether to apply `ln(1 + x)` after size-factor
///   normalisation.
/// * `mean_center` - Whether to implicitly centre columns during SVD.
/// * `normalise_variance` - Whether to implicitly scale columns to unit
///   variance during SVD.
/// * `no_pcs` - Number of principal components to compute.
/// * `random_svd` - If true, use randomised sparse SVD; otherwise use
///   Lanczos-based sparse SVD.
/// * `seed` - Seed for reproducibility (randomised SVD and Lanczos init).
/// * `verbose` - Controls verbosity of timing output.
///
/// ### Returns
///
/// A `ScrubletPcaRes` tuple of `(scores, loadings, gene_means, gene_stds)`
/// where scores is `n_cells x no_pcs`, loadings is `n_genes x no_pcs`,
/// and means/stds are per-gene vectors for downstream doublet projection.
#[allow(clippy::too_many_arguments)]
pub fn pca_scrublet(
    f_path_gene: &str,
    cell_indices: &[usize],
    gene_indices: &[usize],
    hvg_library_sizes: &[usize],
    target_size: f32,
    log_transform: bool,
    mean_center: bool,
    normalise_variance: bool,
    no_pcs: usize,
    random_svd: bool,
    seed: usize,
    verbose: bool,
) -> ScrubletPcaRes {
    let start_total = Instant::now();
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();
    let n_cells = cell_indices.len();

    let start_reading = Instant::now();
    let reader = ParallelSparseReader::new(f_path_gene).unwrap();
    let mut gene_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(gene_indices);
    if verbose {
        println!("Loaded in data: {:.2?}", start_reading.elapsed());
    }

    let start_prep = Instant::now();

    // Filter to selected cells -- after this, chunk.indices are 0-based
    // positions into cell_indices (same ordering as hvg_library_sizes).
    gene_chunks.par_iter_mut().for_each(|chunk| {
        chunk.filter_selected_cells(&cell_set);
    });

    // Re-normalise using HVG library sizes instead of the baked-in total
    // library size normalisation from disk.
    gene_chunks.par_iter_mut().for_each(|chunk| {
        for (i, &pos) in chunk.indices.iter().enumerate() {
            let raw_count = chunk.data_raw.get(i) as f32;
            let lib_size = hvg_library_sizes[pos as usize] as f32;
            let val = if log_transform {
                ((raw_count / lib_size) * target_size).ln_1p()
            } else {
                (raw_count / lib_size) * target_size
            };
            chunk.data_norm[i] = F16::from(half::f16::from_f32(val.clamp(-65504.0, 65504.0)));
        }
    });

    // Assemble into CSC -- remains sparse throughout
    let csc = from_gene_chunks::<f32>(&gene_chunks, n_cells);
    drop(gene_chunks);

    // Column statistics for (a) implicit centering/scaling in SVD and
    // (b) downstream projection of simulated doublets
    let col_means = sparse_csc_column_means(&csc, true);
    let col_stds = sparse_csc_column_stds(&csc, &col_means, true);

    let means_for_svd = if mean_center {
        Some(&col_means[..])
    } else {
        None
    };
    let stds_for_svd = if normalise_variance {
        Some(&col_stds[..])
    } else {
        None
    };

    if verbose {
        println!("Finished data preparation: {:.2?}", start_prep.elapsed());
    }

    let start_svd = Instant::now();

    let (scores, loadings) = if random_svd {
        let svd_res = randomised_sparse_svd::<f32, f32>(
            &csc,
            no_pcs,
            seed as u64,
            true,
            Some(100_usize),
            None,
            means_for_svd,
            stds_for_svd,
        );
        let scores = compute_pc_scores(&svd_res);
        let scores = scores.submatrix(0, 0, n_cells, no_pcs).to_owned();
        let loadings = svd_res
            .v()
            .submatrix(0, 0, gene_indices.len(), no_pcs)
            .to_owned();
        (scores, loadings)
    } else {
        let svd_res = sparse_svd_lanczos::<f32, f32, f32>(
            &csc,
            no_pcs,
            seed as u64,
            true,
            means_for_svd,
            stds_for_svd,
        );
        let scores = compute_pc_scores(&svd_res);
        let loadings = svd_res
            .v()
            .submatrix(0, 0, gene_indices.len(), no_pcs)
            .to_owned();
        (scores, loadings)
    };

    if verbose {
        println!("Finished PCA calculations: {:.2?}", start_svd.elapsed());
        println!(
            "Total run time PCA detection: {:.2?}",
            start_total.elapsed()
        );
    }

    (scores, loadings, col_means, col_stds)
}

/// Find the doublet score threshold using Otsu's method
///
/// Determines the optimal threshold for separating singlets from doublets
/// by maximising the between-class variance of the observed score
/// distribution. This is equivalent to minimising the weighted within-class
/// variance across all possible thresholds.
///
/// Otsu's method is robust to both cleanly bimodal distributions (two
/// separated peaks) and skewed distributions (sharp singlet peak with a
/// long doublet tail), because it does not rely on peak or valley detection.
/// It simply finds the split point that best separates the scores into two
/// groups with distinct means.
///
/// The algorithm proceeds as follows:
///
/// 1. Build a normalised histogram of observed scores with `n_bins` bins.
/// 2. For each candidate threshold (bin boundary), compute the between-class
///    variance: `w0 * w1 * (mu0 - mu1)^2`, where `w0`/`w1` are the class
///    weights and `mu0`/`mu1` are the class means.
/// 3. Return the bin centre that maximises this quantity.
///
/// ### Params
///
/// * `scores_obs` - Doublet scores for observed cells.
/// * `n_bins` - Number of histogram bins (50-100 works well in practice).
///
/// ### Returns
///
/// The score threshold. Cells with scores above this value should be
/// called as doublets.
fn find_threshold_otsu(scores_obs: &[f32], n_bins: usize) -> f32 {
    let (max_score, min_score) = array_max_min(scores_obs);

    if (max_score - min_score).abs() < 1e-6 {
        return (min_score + max_score) / 2.0;
    }

    let bin_width = (max_score - min_score) / n_bins as f32;
    let mut hist = vec![0usize; n_bins];

    for &score in scores_obs {
        let bin = ((score - min_score) / bin_width).floor() as usize;
        hist[bin.min(n_bins - 1)] += 1;
    }

    let total = scores_obs.len() as f32;
    let prob: Vec<f32> = hist.iter().map(|&c| c as f32 / total).collect();

    // Precompute cumulative sums
    let mut w0 = 0.0f32;
    let mut sum0 = 0.0f32;
    let total_mean: f32 = prob.iter().enumerate().map(|(i, &p)| i as f32 * p).sum();

    let mut best_variance = 0.0f32;
    let mut best_bin = 0usize;

    for i in 0..n_bins {
        w0 += prob[i];
        if w0 < 1e-10 {
            continue;
        }

        let w1 = 1.0 - w0;
        if w1 < 1e-10 {
            break;
        }

        sum0 += i as f32 * prob[i];

        let mu0 = sum0 / w0;
        let mu1 = (total_mean - sum0) / w1;

        let between_var = w0 * w1 * (mu0 - mu1).powi(2);

        if between_var > best_variance {
            best_variance = between_var;
            best_bin = i;
        }
    }

    min_score + (best_bin as f32 + 0.5) * bin_width
}

/// Find threshold between singlets and doublets using combined score
/// distribution
///
/// Creates a histogram of both observed and simulated scores, then finds
/// the valley (minimum) between the two peaks (singlets vs doublets).
///
/// ### Params
///
/// * `scores_obs` - Doublet scores from observed cells (mostly singlets)
/// * `scores_sim` - Doublet scores from simulated doublets
/// * `n_bins` - Number of histogram bins
///
/// ### Returns
///
/// Threshold score at the valley between the two modes
fn find_threshold_min(scores_obs: &[f32], scores_sim: &[f32], n_bins: usize) -> f32 {
    let mut all_scores: Vec<f32> = Vec::with_capacity(scores_obs.len() + scores_sim.len());
    all_scores.extend_from_slice(scores_obs);
    all_scores.extend_from_slice(scores_sim);

    let (max_score, min_score) = array_max_min(&all_scores);

    if (max_score - min_score).abs() < 1e-6 {
        return (min_score + max_score) / 2.0;
    }

    let bin_width = (max_score - min_score) / n_bins as f32;
    let mut hist = vec![0usize; n_bins];

    for &score in &all_scores {
        let bin = ((score - min_score) / bin_width).floor() as usize;
        hist[bin.min(n_bins - 1)] += 1;
    }

    let smoothed: Vec<f32> = moving_average(&hist, 5)
        .into_iter()
        .map(|x| x as f32)
        .collect();

    // find all local maxima
    let mut peaks: Vec<(usize, f32)> = Vec::new();
    for i in 1..(smoothed.len() - 1) {
        if smoothed[i] >= smoothed[i - 1] && smoothed[i] >= smoothed[i + 1] && smoothed[i] > 0.0 {
            peaks.push((i, smoothed[i]));
        }
    }

    if peaks.len() < 2 {
        // No clear bimodality -- fall back to median of simulated scores
        let mut sorted_sim = scores_sim.to_vec();
        sorted_sim.sort_by(|a, b| a.partial_cmp(b).unwrap());
        return sorted_sim[sorted_sim.len() / 4]; // Q1 of simulated as conservative threshold
    }

    // Take the two tallest peaks
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let peak_a = peaks[0].0.min(peaks[1].0);
    let peak_b = peaks[0].0.max(peaks[1].0);

    // Find the minimum between them
    let mut min_idx = peak_a;
    let mut min_val = smoothed[peak_a];
    for i in peak_a..=peak_b {
        if smoothed[i] < min_val {
            min_val = smoothed[i];
            min_idx = i;
        }
    }

    min_score + (min_idx as f32 + 0.5) * bin_width
}

/// Apply moving average smoothing to histogram data
///
/// Smooths a histogram by averaging each bin with its neighbours. This reduces
/// noise and makes peak/valley detection more robust.
///
/// ### Params
///
/// * `data` - Histogram bin counts
/// * `window` - Size of the moving average window (e.g., 3 means average with
///   1 neighbor on each side)
///
/// ### Returns
///
/// Smoothed histogram with the same length as input
///
/// ### Example
///
/// With window = 3, each bin is averaged with its immediate neighbors:
///
/// ```text
/// Input:  [1, 5, 2, 8, 3]
/// Output: [3, 3, 5, 4, 5]  // (1+5+2)/3, (1+5+2)/3, (5+2+8)/3, (2+8+3)/3, (8+3)/2
/// ```
fn moving_average(data: &[usize], window: usize) -> Vec<usize> {
    let half_window = window / 2;
    data.iter()
        .enumerate()
        .map(|(i, _)| {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(data.len());
            let sum: usize = data[start..end].iter().sum();
            sum / (end - start)
        })
        .collect()
}

////////////////////
// Main structure //
////////////////////

/// Structure for Scrublet algorithm
///
/// ### Fields
///
/// * `f_path_gene` - Path to the binarised file in CSC format.
/// * `f_path_cell` - Path to the binarised file in CSR format.
/// * `params` - The Scrublet parameters
/// * `n_cells` - Number of observed cells.
/// * `n_cells_sim` - Number of simulated cells.
/// * `cells_to_keep` - Indices of cells to keep/include in this analysis
#[derive(Clone, Debug)]
pub struct Scrublet {
    f_path_gene: String,
    f_path_cell: String,
    params: ScrubletParams,
    n_cells: usize,
    n_cells_sim: usize,
    cells_to_keep: Vec<usize>,
    hvg_library_sizes: Vec<usize>,
}

impl Scrublet {
    /// Generate a new instance
    ///
    /// ### Params
    ///
    /// * `f_path_gene` - Path to the binarised file in CSC format.
    /// * `f_path_cell` - Path to the binarised file in CSR format.
    /// * `params` - The Scrublet parameters to use.
    /// * `cell_indices` - Slice of usizes indicating which cells to keep/use.
    pub fn new(
        f_path_gene: &str,
        f_path_cells: &str,
        params: ScrubletParams,
        cell_indices: &[usize],
    ) -> Self {
        let n_cells = cell_indices.len();

        Scrublet {
            f_path_gene: f_path_gene.to_string(),
            f_path_cell: f_path_cells.to_string(),
            params,
            n_cells,
            n_cells_sim: 0,
            cells_to_keep: cell_indices.to_vec(),
            hvg_library_sizes: Vec::new(),
        }
    }

    /// Main function to run Scrublet
    ///
    /// ### Params
    ///
    /// * `streaming` - Shall the data be streamed. Reduces memory pressure
    ///   during HVG detection.
    /// * `manual_threshold` - Optional threshold for when to call doublets. If
    ///   not provided, will be estimated from the data.
    /// * `n_bins` - Number of bins to use for histogram construction
    ///   (typically 50 - 100).
    /// * `seed` - Seed for reproducibility.
    /// * `verbose` - Controls verbosity of the function
    ///
    /// ### Returns
    ///
    /// Initialised self.
    pub fn run_scrublet(
        &mut self,
        streaming: bool,
        seed: usize,
        verbose: bool,
        return_combined_pca: bool,
        return_pairs: bool,
    ) -> FinalScrubletRes {
        if verbose {
            println!("Identifying highly variable genes...");
        }
        let start_all = Instant::now();
        let start_hvg = Instant::now();

        let hvg_genes = self.get_hvg(streaming, verbose);

        let end_hvg = start_hvg.elapsed();
        if verbose {
            println!(
                "Using {} highly variable genes. Done in {:.2?}",
                hvg_genes.len(),
                end_hvg
            );
        }

        let cell_reader = ParallelSparseReader::new(&self.f_path_cell).unwrap();
        let hvg_set: FxHashSet<usize> = hvg_genes.iter().copied().collect();

        let hvg_library_sizes: Vec<usize> = self
            .cells_to_keep
            .par_iter()
            .map(|&cell_idx| {
                let chunk = cell_reader.read_cell(cell_idx);
                chunk
                    .indices
                    .iter()
                    .enumerate()
                    .filter(|(_, gene_idx)| hvg_set.contains(&(**gene_idx as usize)))
                    .map(|(i, _)| chunk.data_raw.get(i) as usize)
                    .sum()
            })
            .collect();

        let target_size = self.params.target_size.unwrap_or_else(|| {
            let sum = hvg_library_sizes.iter().sum::<usize>() as f32;
            sum / hvg_library_sizes.len() as f32
        });

        self.hvg_library_sizes = hvg_library_sizes;

        if verbose {
            println!("Simulating doublets...");
        }
        let start_doublet_gen = Instant::now();

        let (sim_chunks, pair_1, pair_2) = self.simulate_doublets(target_size, &hvg_genes, seed);

        let end_doublet_gen = start_doublet_gen.elapsed();
        if verbose {
            println!(
                "Simulated {} doublets. Done in {:.2?}",
                sim_chunks.len(),
                end_doublet_gen
            );
        }

        if verbose {
            println!("Running PCA...");
        }
        let start_pca = Instant::now();

        let combined_pca = self.run_pca(&sim_chunks, &hvg_genes, target_size, verbose, seed);

        let end_pca = start_pca.elapsed();
        if verbose {
            println!("Done with PCA in {:.2?}", end_pca);
        }

        if verbose {
            println!("Building kNN graph...");
        }
        let start_knn = Instant::now();

        let knn_indices = self
            .build_combined_knn(combined_pca.as_ref(), seed, verbose)
            .expect("Failed to build kNN graph");

        let end_knn = start_knn.elapsed();
        if verbose {
            println!("Done with KNN generation in {:.2?}", end_knn);
        }

        if verbose {
            println!("Calculating doublet scores...");
        }
        let start_doublets = Instant::now();

        let doublet_scores: ScrubletDoubletScores = self.calculate_doublet_scores(&knn_indices);

        let res = self.call_doublets(
            doublet_scores,
            self.params.manual_threshold,
            self.params.n_bins,
            verbose,
        );

        let end_doublets = start_doublets.elapsed();
        if verbose {
            println!("Done with doublet scoring and calling {:.2?}", end_doublets);
        }

        let end_all = start_all.elapsed();
        if verbose {
            println!("Finished Scrublet {:.2?}", end_all);
        }

        let pca_out = if return_combined_pca {
            Some(combined_pca)
        } else {
            None
        };
        let pair_1_out = if return_pairs { Some(pair_1) } else { None };
        let pair_2_out = if return_pairs { Some(pair_2) } else { None };

        (res, pca_out, pair_1_out, pair_2_out)
    }

    /// Get the indices of the highly variable genes
    ///
    /// Identify the HVGs for subsequent PCA.
    ///
    /// ### Params
    ///
    /// * `streaming` - Shall the data be loaded in a streaming fashion. Reduces
    ///   memory pressure
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Vector of indices of the HVG.
    fn get_hvg(&self, streaming: bool, verbose: bool) -> Vec<usize> {
        let hvg_type = parse_hvg_method(&self.params.hvg_method)
            .ok_or_else(|| format!("Invalid HVG method: {}", &self.params.hvg_method))
            .unwrap();

        let hvg_res: HvgRes = if streaming {
            match hvg_type {
                HvgMethod::Vst => get_hvg_vst_streaming(
                    &self.f_path_gene,
                    &self.cells_to_keep,
                    self.params.loess_span as f32,
                    self.params.clip_max,
                    verbose,
                ),
                HvgMethod::MeanVarBin => get_hvg_mvb_streaming(),
                HvgMethod::Dispersion => get_hvg_dispersion_streaming(),
            }
        } else {
            match hvg_type {
                HvgMethod::Vst => get_hvg_vst(
                    &self.f_path_gene,
                    &self.cells_to_keep,
                    self.params.loess_span as f32,
                    self.params.clip_max,
                    verbose,
                ),
                HvgMethod::MeanVarBin => get_hvg_mvb(),
                HvgMethod::Dispersion => get_hvg_dispersion(),
            }
        };

        let n_genes = hvg_res.mean.len() as f32;
        let n_genes_to_take = (n_genes * (1.0 - self.params.min_gene_var_pctl)).ceil() as usize;

        let mut indices: Vec<(usize, f64)> = hvg_res
            .var_std
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indices.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indices.truncate(n_genes_to_take);

        let mut result: Vec<usize> = indices.into_iter().map(|(i, _)| i).collect();
        result.sort_unstable();

        result
    }

    /// Simulate doublets
    ///
    /// Generate artifical doublets based on the real data. The algorithm
    /// will only construct doublets from the
    ///
    /// ### Params
    ///
    /// * `hvg_genes` - Indices of the highly variable genes.
    /// * `target_size` - The library target size. Needs to be the same as the
    ///   original one (in single cell typicall `1e4`).
    /// * `seed` - Seed for reproducibility.
    ///
    /// ### Returns
    ///
    /// A `Vec<CsrCellChunk>` that contains the artificially created doublets.
    fn simulate_doublets(
        &mut self,
        target_size: f32,
        hvg_genes: &[usize],
        seed: usize,
    ) -> (Vec<CsrCellChunk>, Vec<usize>, Vec<usize>) {
        let n_sim_doublets = (self.n_cells as f32 * self.params.sim_doublet_ratio) as usize;
        self.n_cells_sim = n_sim_doublets;
        let mut rng = StdRng::seed_from_u64(seed as u64);

        let pairs: Vec<(usize, usize)> = (0..n_sim_doublets)
            .map(|_| {
                let pos_i = rng.random_range(0..self.cells_to_keep.len());
                let pos_j = rng.random_range(0..self.cells_to_keep.len());
                (pos_i, pos_j)
            })
            .collect();

        let hvg_set: FxHashSet<usize> = hvg_genes.iter().copied().collect();
        let gene_to_hvg_idx: FxHashMap<usize, u32> = hvg_genes
            .iter()
            .enumerate()
            .map(|(hvg_idx, &orig_idx)| (orig_idx, hvg_idx as u32))
            .collect();

        let reader = ParallelSparseReader::new(&self.f_path_cell).unwrap();

        let doublets: Vec<CsrCellChunk> = pairs
            .par_iter()
            .enumerate()
            .map(|(doublet_idx, &(pos_i, pos_j))| {
                let cell_idx_i = self.cells_to_keep[pos_i];
                let cell_idx_j = self.cells_to_keep[pos_j];

                let cell1 = reader.read_cell(cell_idx_i);
                let cell2 = reader.read_cell(cell_idx_j);

                let hvg_combined_lib_size =
                    self.hvg_library_sizes[pos_i] + self.hvg_library_sizes[pos_j];

                let mut doublet = CsrCellChunk::add_cells_scrublet(
                    &cell1,
                    &cell2,
                    &hvg_set,
                    hvg_combined_lib_size,
                    target_size,
                    self.params.log_transform,
                    doublet_idx,
                );

                for idx in doublet.indices.iter_mut() {
                    *idx = gene_to_hvg_idx[&(*idx as usize)];
                }

                doublet
            })
            .collect();

        let (pair_1, pair_2): (Vec<usize>, Vec<usize>) = pairs.into_iter().unzip();

        (doublets, pair_1, pair_2)
    }

    /// Run PCA
    ///
    /// Runs PCA prior to kNN construction.
    ///
    /// ### Params
    ///
    /// * `sim_chunks` - Slice of `CsrCellChunk` containing the simulated
    ///   doublets.
    /// * `hvg_genes` - The indices of the highly variable genes.
    /// * `verbose` - Controls verbosity of the function.
    /// * `seed` - Seed for reproducibility. Relevant when using randomised
    ///   SVD.
    ///
    /// ### Returns
    ///
    /// The PCA scores with the top rows representing the actual data and
    /// the bottom rows the simulated data.
    fn run_pca(
        &self,
        sim_chunks: &[CsrCellChunk],
        hvg_genes: &[usize],
        target_size: f32,
        verbose: bool,
        seed: usize,
    ) -> Mat<f32> {
        let pca_res: ScrubletPcaRes = pca_scrublet(
            &self.f_path_gene,
            &self.cells_to_keep,
            hvg_genes,
            &self.hvg_library_sizes,
            target_size,
            self.params.log_transform,
            self.params.mean_center,
            self.params.normalise_variance,
            self.params.no_pcs,
            self.params.random_svd,
            seed,
            verbose,
        );

        let scaled_sim = scale_cell_chunks_with_stats(
            sim_chunks,
            &pca_res.2,
            &pca_res.3,
            self.params.mean_center,
            self.params.normalise_variance,
            hvg_genes.len(),
        );

        let pca_sim = &scaled_sim * pca_res.1;

        let combined = concat![[pca_res.0], [pca_sim]];

        combined
    }

    /// Generate the kNN graph
    ///
    /// ### Params
    ///
    /// * `embd` - The embedding matrix to use for the generation of the kNN
    ///   graph. Usually the PCA of observed and simulated doublet cells.
    /// * `seed` - Seed for reproducibility. Relevant when using randomised
    /// * `verbose` - Controls verbosity of the function.
    ///
    /// ### Returns
    ///
    /// The kNN graph as a `Vec<Vec<usize>>`.
    fn build_combined_knn(
        &self,
        embd: MatRef<f32>,
        seed: usize,
        verbose: bool,
    ) -> Result<Vec<Vec<usize>>, String> {
        let knn_method = parse_knn_method(&self.params.knn_params.knn_method).unwrap_or_default();

        let k_adj = self.calculate_k_adj();

        if verbose {
            println!("Using {} neighbours in the kNN generation.", k_adj);
        }

        let knn = match knn_method {
            KnnSearch::Hnsw => generate_knn_hnsw(
                embd.as_ref(),
                &self.params.knn_params.ann_dist,
                k_adj,
                self.params.knn_params.m,
                self.params.knn_params.ef_construction,
                self.params.knn_params.ef_search,
                seed,
                verbose,
            ),
            KnnSearch::Annoy => generate_knn_annoy(
                embd.as_ref(),
                &self.params.knn_params.ann_dist,
                k_adj,
                self.params.knn_params.n_tree,
                self.params.knn_params.search_budget,
                seed,
                verbose,
            ),
            KnnSearch::NNDescent => generate_knn_nndescent(
                embd.as_ref(),
                &self.params.knn_params.ann_dist,
                k_adj,
                self.params.knn_params.diversify_prob,
                self.params.knn_params.ef_budget,
                self.params.knn_params.delta,
                seed,
                verbose,
            ),
            KnnSearch::Exhaustive => generate_knn_exhaustive(
                embd.as_ref(),
                &self.params.knn_params.ann_dist,
                k_adj,
                verbose,
            ),
            KnnSearch::Ivf => generate_knn_ivf(
                embd.as_ref(),
                &self.params.knn_params.ann_dist,
                k_adj,
                self.params.knn_params.n_list,
                self.params.knn_params.n_list,
                seed,
                verbose,
            ),
        };

        Ok(knn)
    }

    /// Calculate the doublet scores
    ///
    /// ### Params
    ///
    /// * `knn_indices` - A slice of `Vec<usize>` indicating the nearest
    ///   neighbours.
    ///
    /// ### Returns
    ///
    /// `ScrubletDoubletScores` type alias that represents the scores and errors
    /// of the observed and simulated cells.
    fn calculate_doublet_scores(&self, knn_indices: &[Vec<usize>]) -> ScrubletDoubletScores {
        let n_obs = self.n_cells;
        let n_sim = self.n_cells_sim;

        let r = n_sim as f32 / n_obs as f32;
        let rho = self.params.expected_doublet_rate;
        let se_rho = self.params.stdev_doublet_rate;

        let k_adj = self.calculate_k_adj();
        let n_adj = k_adj as f32;

        let scores_errors: Vec<(f32, f32)> = knn_indices
            .par_iter()
            .map(|neighbours| {
                let n_sim_neigh = neighbours.iter().filter(|&&idx| idx >= n_obs).count() as f32;
                let q = (n_sim_neigh + 1.0) / (n_adj + 2.0);
                let denominator = 1.0 - rho - q * (1.0 - rho - rho / r);
                let score = if denominator.abs() > 1e-10 {
                    (q * rho / r) / denominator
                } else {
                    0.0
                };

                let se_q = (q * (1.0 - q) / (n_adj + 3.0)).sqrt();
                let factor = q * rho / r / (denominator * denominator);
                let se_score = factor
                    * ((se_q / q * (1.0 - rho)).powi(2) + (se_rho / rho * (1.0 - q)).powi(2))
                        .sqrt();

                (score.max(0.0), se_score.max(1e-10))
            })
            .collect();

        let (scores_obs, errors_obs): (Vec<f32>, Vec<f32>) =
            scores_errors[..n_obs].iter().copied().unzip();

        let (scores_sim, errors_sim): (Vec<f32>, Vec<f32>) =
            scores_errors[n_obs..].iter().copied().unzip();

        (scores_obs, errors_obs, scores_sim, errors_sim)
    }

    /// Call the doublets
    ///
    /// ### Params
    ///
    /// * `doublet_scores` - type alias that represents the scores and errors
    ///   of the observed and simulated cells.
    /// * `manual_threshold` - Optional manual threshold for when to call a
    ///   singlet or doublet.
    /// * `n_bins` - Number of bins to use.
    /// * `verbose` - Controls verbosity of the function
    ///
    /// ### Returns
    ///
    /// `ScrubletResult` with the final results of the algorithm.
    fn call_doublets(
        &self,
        doublet_scores: ScrubletDoubletScores,
        manual_threshold: Option<f32>,
        n_bins: usize,
        verbose: bool,
    ) -> ScrubletResult {
        let threshold = manual_threshold.unwrap_or_else(|| {
            let t = find_threshold_otsu(&doublet_scores.0, n_bins);
            if verbose {
                println!("Automatically set threshold at doublet score = {:.4}", t);
            }
            t
        });

        let predicted_doublets: Vec<bool> = doublet_scores
            .0
            .iter()
            .map(|&score| score > threshold)
            .collect();

        let z_scores: Vec<f32> = doublet_scores
            .0
            .iter()
            .zip(doublet_scores.1.iter())
            .map(|(&score, &error)| (score - threshold) / error)
            .collect();

        let n_detected = predicted_doublets.iter().filter(|&&x| x).count();
        let detected_doublet_rate = n_detected as f32 / doublet_scores.0.len() as f32;

        let n_detectable = doublet_scores.2.iter().filter(|&&s| s > threshold).count();
        let detectable_doublet_fraction = n_detectable as f32 / doublet_scores.2.len() as f32;

        let overall_doublet_rate = if detectable_doublet_fraction > 0.01 {
            detected_doublet_rate / detectable_doublet_fraction
        } else {
            0.0
        };

        if verbose {
            println!(
                "Detected doublet rate = {:.1}%",
                100.0 * detected_doublet_rate
            );
            println!(
                "Estimated detectable doublet fraction = {:.1}%",
                100.0 * detectable_doublet_fraction
            );
            println!("Overall doublet rate:");
            println!(
                "  Expected  = {:.1}%",
                100.0 * self.params.expected_doublet_rate
            );
            println!("  Estimated = {:.1}%", 100.0 * overall_doublet_rate);
        };

        ScrubletResult {
            predicted_doublets,
            doublet_scores_obs: doublet_scores.0,
            doublet_scores_sim: doublet_scores.2,
            doublet_errors_obs: doublet_scores.1,
            z_scores,
            threshold,
            detected_doublet_rate,
            detectable_doublet_fraction,
            overall_doublet_rate,
        }
    }

    /// Calculates the adjusted k based on number actual cells and simulated
    /// cells
    fn calculate_k_adj(&self) -> usize {
        let k = if self.params.knn_params.k == 0 {
            ((self.n_cells as f32).sqrt() * 0.5).round() as usize
        } else {
            self.params.knn_params.k
        };

        let r = self.n_cells_sim as f32 / self.n_cells as f32;
        (k as f32 * (1.0 + r)).round() as usize
    }
}
