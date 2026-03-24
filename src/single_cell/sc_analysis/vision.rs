//! Implementation of the VISION framework to score spatially correlated
//! gene sets. See DeTamaso, et al., Nat. Commun., 2019

use extendr_api::{Conversions, List};
use rayon::prelude::*;
use std::time::Instant;

use crate::core::math::vector_helpers::rank_vector;
use crate::prelude::*;

////////////
// VISION //
////////////

/// Structure to store the indices of the SignatureGenes
#[derive(Clone, Debug)]
pub struct SignatureGenes {
    /// The gene indices of the positive genes
    pub positive: Vec<usize>,
    /// The gene indices of the negative genes
    pub negative: Vec<usize>,
}

/// Helper function to transform an R gene set list to `Vec<SignatureGenes>`
///
/// ### Params
///
/// * `gs_list` - Initial R list with the gene sets for VISION
///
/// ### Returns
///
/// The vector of SignatureGenes
pub fn r_list_to_sig_genes(gs_list: List) -> extendr_api::Result<Vec<SignatureGenes>> {
    let mut gene_signatures: Vec<SignatureGenes> = Vec::with_capacity(gs_list.len());

    for i in 0..gs_list.len() {
        let r_obj = gs_list.elt(i)?;
        let gs_list_i = r_obj.as_list().ok_or_else(|| {
            extendr_api::Error::from(
                "The lists in the gs_list could not be converted. Please check!",
            )
        })?;
        gene_signatures.push(SignatureGenes::from_r_list(gs_list_i));
    }

    Ok(gene_signatures)
}

/////////////
// Helpers //
/////////////

/// Calculate VISION signature scores for a single cell
///
/// ### Params
///
/// * `cell` - The CsrCellChunk
/// * `signatures` - Slice of `SignatureGenes` to calculate the scores for
/// * `total_genes` - Total number of represented genes
///
/// ### Returns
///
/// A Vec<f32> with a score for each of the `SignatureGenes` that were supplied
fn calculate_vision_scores_for_cell(
    cell: &CsrCellChunk,
    signatures: &[SignatureGenes],
    total_genes: usize,
) -> Vec<f32> {
    // helper
    let get_expr = |gene_idx: usize| -> f32 {
        match cell.indices.binary_search(&(gene_idx as u32)) {
            Ok(pos) => cell.data_norm[pos].to_f32(),
            Err(_) => 0.0,
        }
    };

    // Cell-level statistics (ALL genes including zeros)
    let sum: f32 = cell.data_norm.iter().map(|x| x.to_f32()).sum();
    let sum_sq: f32 = cell
        .data_norm
        .iter()
        .map(|x| {
            let v = x.to_f32();
            v * v
        })
        .sum();

    let mu_cell = sum / total_genes as f32;
    let var_cell = (sum_sq / total_genes as f32) - (mu_cell * mu_cell);
    let sigma_cell = var_cell.sqrt();

    // Score signatures
    signatures
        .iter()
        .map(|sig| {
            let sum_pos: f32 = sig.positive.iter().map(|&idx| get_expr(idx)).sum();
            let sum_neg: f32 = sig.negative.iter().map(|&idx| get_expr(idx)).sum();

            let signature_size = (sig.positive.len() + sig.negative.len()) as f32;

            if signature_size == 0.0 || sigma_cell == 0.0 {
                return 0.0;
            }

            let mean_sig = (sum_pos - sum_neg) / signature_size;

            // R "znorm_columns" formula
            (mean_sig - mu_cell) / sigma_cell
        })
        .collect()
}

/// Calculate Geary's C for a single signature (pathway)
///
/// This implements a modified approach akin to VISION.
///
/// ### Params
///
/// * `scores` - Ranked signature scores for all cells
/// * `knn_indices` - KNN indices matrix (cells x k)
/// * `knn_weights` - KNN weights matrix (cells x k)
///
/// ### Returns
///
/// Geary's C statistic
fn geary_c(scores: &[f64], knn_indices: &[Vec<usize>], knn_weights: &[Vec<f32>]) -> f64 {
    let n = scores.len();

    let mean: f64 = scores.iter().sum::<f64>() / n as f64;
    let variance: f64 = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>();

    if variance == 0.0 {
        return 0.0;
    }

    let mut numerator = 0.0;
    let mut total_weight = 0.0;

    // unsafe unchecked access in hot loop
    for (i, (indices, weights)) in knn_indices.iter().zip(knn_weights.iter()).enumerate() {
        let xi = unsafe { *scores.get_unchecked(i) };
        for (&j, &w) in indices.iter().zip(weights.iter()) {
            let xj = unsafe { *scores.get_unchecked(j) };
            numerator += w as f64 * (xi - xj).powi(2);
            total_weight += w as f64;
        }
    }

    let norm = 2.0 * total_weight * variance / (n as f64 - 1.0);

    numerator / norm
}

/// Calculate KNN weights using exponential kernel
///
/// ### Params
///
/// * `knn_indices` - KNN indices (cells x k)
/// * `knn_distances` - KNN squared distances (cells x k) - use Euclidean here!
///
/// ### Returns
///
/// KNN weights matrix (cells x k)
fn calc_knn_weights(knn_indices: &[Vec<usize>], knn_distances: &[Vec<f32>]) -> Vec<Vec<f32>> {
    knn_indices
        .par_iter() // Parallel iterator
        .zip(knn_distances.par_iter())
        .map(|(indices, distances)| {
            if distances.is_empty() {
                return vec![];
            }

            let sigma_sq = distances.last().copied().unwrap_or(1.0);

            if sigma_sq == 0.0 {
                return vec![1.0; indices.len()];
            }

            distances
                .iter()
                .map(|&d_sq| (-d_sq / sigma_sq).exp())
                .collect()
        })
        .collect()
}

//////////
// Main //
//////////

/// Calculate VISION signature scores across a set of cells
///
/// ### Params
///
/// * `f_path` -  File path to the cell-based binary file.
/// * `signatures` - Slice of `SignatureGenes` to calculate the scores for
/// * `cells_to_keep` - Vector of indices with the cells to keep.
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// A `Vec<Vec<f32>>` with cells x scores per gene set (pair)
pub fn calculate_vision(
    f_path: &str,
    gene_signs: &[SignatureGenes],
    cells_to_keep: &[usize],
    verbose: bool,
) -> Vec<Vec<f32>> {
    let start_read = Instant::now();
    let reader = ParallelSparseReader::new(f_path).unwrap();
    let no_genes = reader.get_header().total_genes;
    let cell_chunks: Vec<CsrCellChunk> = reader.read_cells_parallel(cells_to_keep);
    let end_read = start_read.elapsed();

    if verbose {
        println!("Loaded in data: {:.2?}", end_read);
    }

    let start_signatures = Instant::now();
    let signature_scores: Vec<Vec<f32>> = cell_chunks
        .par_iter()
        .map(|chunk| calculate_vision_scores_for_cell(chunk, gene_signs, no_genes))
        .collect();
    let end_signatures = start_signatures.elapsed();

    if verbose {
        println!("Calculated VISION scores: {:.2?}", end_signatures);
    }

    signature_scores // cells x signatures
}

/// Calculate VISION signature scores across a set of cells (streaming)
///
/// The streaming version of the function.
///
/// ### Params
///
/// * `f_path` -  File path to the cell-based binary file.
/// * `signatures` - Slice of `SignatureGenes` to calculate the scores for
/// * `cells_to_keep` - Vector of indices with the cells to keep.
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// A `Vec<Vec<f32>>` with cells x scores per gene set (pair)
pub fn calculate_vision_streaming(
    f_path: &str,
    gene_signs: &[SignatureGenes],
    cells_to_keep: &[usize],
    verbose: bool,
) -> Vec<Vec<f32>> {
    const CHUNK_SIZE: usize = 50000;

    let total_chunks = cells_to_keep.len().div_ceil(CHUNK_SIZE);
    let reader = ParallelSparseReader::new(f_path).unwrap();
    let no_genes = reader.get_header().total_genes;

    let mut all_results: Vec<Vec<f32>> = Vec::with_capacity(cells_to_keep.len());

    for (chunk_idx, cell_indices_chunk) in cells_to_keep.chunks(CHUNK_SIZE).enumerate() {
        let start_chunk = Instant::now();

        let cell_chunks = reader.read_cells_parallel(cell_indices_chunk);

        let chunk_scores: Vec<Vec<f32>> = cell_chunks
            .par_iter()
            .map(|chunk| calculate_vision_scores_for_cell(chunk, gene_signs, no_genes))
            .collect();

        all_results.extend(chunk_scores);

        if verbose {
            let elapsed = start_chunk.elapsed();
            let pct_complete = ((chunk_idx + 1) as f32 / total_chunks as f32) * 100.0;
            println!(
                "Processing chunk {} out of {} (took {:.2?}, completed {:.1}%)",
                chunk_idx + 1,
                total_chunks,
                elapsed,
                pct_complete
            );
        }
    }

    all_results
}

/// Calculate VISION local autocorrelation scores
///
/// ### Params
///
/// * `pathway_scores` - Vector representing the actual vision scores:
///   `cells x pathways
/// * `random_scores_by_cluster` - Vector representing the random scores by
///   cluster: `clusters -> (cells x sigs)`
/// * `cluster_membership` - Vector representing to which cluster a given
///   gene set belongs.
/// * `knn_indices` - KNN indices from embedding (cells x k)
/// * `knn_distances` - KNN squared distances (cells x k)
/// * `verbose` - Print progress
///
/// ### Returns
///
/// Tuple of (consistency_scores, p_values) for each pathway
pub fn calc_autocorr_with_clusters(
    pathway_scores: &[Vec<f32>],
    random_scores_by_cluster: &[Vec<Vec<f32>>],
    cluster_membership: &[usize],
    knn_indices: Vec<Vec<usize>>,
    knn_distances: Vec<Vec<f32>>,
    verbose: bool,
) -> (Vec<f64>, Vec<f64>) {
    let start = Instant::now();

    let knn_weights = calc_knn_weights(&knn_indices, &knn_distances);

    if verbose {
        println!("Computed KNN weights: {:.2?}", start.elapsed());
    }

    let n_pathways = pathway_scores[0].len();

    // calculate Geary's C for actual pathways
    let pathway_consistency: Vec<f64> = (0..n_pathways)
        .into_par_iter()
        .map(|pathway_idx| {
            let scores: Vec<f32> = pathway_scores
                .iter()
                .map(|cell| cell[pathway_idx])
                .collect();

            let ranks = rank_vector(&scores);
            let c = geary_c(&ranks.r_float_convert(), &knn_indices, &knn_weights);
            1.0 - c
        })
        .collect();

    if verbose {
        println!("Calculated pathway consistency: {:.2?}", start.elapsed());
    }

    let cluster_bg_consistency: Vec<Vec<f64>> = random_scores_by_cluster
        .iter()
        .map(|cluster_scores| {
            let n_random = cluster_scores[0].len();

            // Single level of parallelism
            (0..n_random)
                .into_par_iter() // Only this one is parallel
                .map(|sig_idx| {
                    let scores: Vec<f32> =
                        cluster_scores.iter().map(|cell| cell[sig_idx]).collect();

                    let ranks = rank_vector(&scores);
                    let c = geary_c(&ranks.r_float_convert(), &knn_indices, &knn_weights);
                    1.0 - c
                })
                .collect()
        })
        .collect();

    // calculate p vals
    let p_vals: Vec<f64> = pathway_consistency
        .iter()
        .enumerate()
        .map(|(i, &fg_c)| {
            let cluster_idx = cluster_membership[i];
            let bg_dist = &cluster_bg_consistency[cluster_idx];

            let n = bg_dist.len();
            let num_greater_equal = bg_dist.iter().filter(|&&bg_c| bg_c >= fg_c).count();

            (num_greater_equal + 1) as f64 / (n + 1) as f64
        })
        .collect();

    if verbose {
        println!("Calculated p-values: {:.2?}", start.elapsed());
    }

    (pathway_consistency, p_vals)
}
