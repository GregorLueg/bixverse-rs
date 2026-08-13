//! Implementation of the VISION framework to score spatially correlated
//! gene sets. See DeTamaso, et al., Nat. Commun., 2019

use extendr_api::{Conversions, List};
use rayon::prelude::*;
use std::time::Instant;

use crate::core::math::vector_helpers::rank_vector;
use crate::prelude::*;
use crate::single_cell::sc_processing::knn::knn_distance_weights;

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
        gene_signatures.push(SignatureGenes::from_r_list(gs_list_i)?);
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

/// Calculate KNN weights using the VISION exponential kernel
///
/// `exp(-d^2 / sigma^2)` with `sigma` the distance to the furthest of the `k`
/// neighbours, row-normalised so every cell contributes the same total weight.
/// That is `computeKNNWeights` in `VISION/R/Projections.R`, and it is
/// [`knn_distance_weights`] at `neighborhood_factor = 1`: the width neighbour
/// is `ceil(k / 1) = k`, i.e. the furthest one.
///
/// The row normalisation matters. [`geary_c`] divides by the total weight, so
/// a global rescaling would cancel, but normalising per row does not: it is
/// what stops cells whose neighbours sit at similar distances from carrying
/// more of the statistic than cells with one close neighbour and a long tail.
///
/// ### Params
///
/// * `knn_distances` - KNN distances (cells x k), ascending
/// * `squared` - `true` when `knn_distances` already holds `d^2`, which is what
///   the `"euclidean"` metric returns, see
///   [`crate::single_cell::sc_processing::knn::distances_are_squared`]
///
/// ### Returns
///
/// KNN weights matrix (cells x k), each row summing to one
///
/// ### References
///
/// DeTomaso, et al., Nat. Commun., 2019
fn calc_knn_weights(knn_distances: &[Vec<f32>], squared: bool) -> Vec<Vec<f32>> {
    knn_distance_weights(knn_distances, 1.0, squared)
}

//////////
// Main //
//////////

/// Calculate VISION signature scores across a set of cells
///
/// ### Params
///
/// * `reader` - Reader over the cell-based count store.
/// * `signatures` - Slice of `SignatureGenes` to calculate the scores for
/// * `cells_to_keep` - Vector of indices with the cells to keep.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A `Vec<Vec<f32>>` with cells x scores per gene set (pair)
pub fn calculate_vision<S: SingleCellReading>(
    reader: &S,
    gene_signs: &[SignatureGenes],
    cells_to_keep: &[usize],
    verbose: usize,
) -> Result<Vec<Vec<f32>>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    let start_read = Instant::now();
    let no_genes = reader.get_header().total_genes;
    let cell_chunks: Vec<CsrCellChunk> = reader.read_cells_parallel(cells_to_keep)?;
    let end_read = start_read.elapsed();

    if verbosity.normal_verbosity() {
        println!("Loaded in data: {:.2?}", end_read);
    }

    let start_signatures = Instant::now();
    let signature_scores: Vec<Vec<f32>> = cell_chunks
        .par_iter()
        .map(|chunk| calculate_vision_scores_for_cell(chunk, gene_signs, no_genes))
        .collect();
    let end_signatures = start_signatures.elapsed();

    if verbosity.normal_verbosity() {
        println!("Calculated VISION scores: {:.2?}", end_signatures);
    }

    // cells x signatures
    Ok(signature_scores)
}

/// Calculate VISION signature scores across a set of cells (streaming)
///
/// The streaming version of the function.
///
/// ### Params
///
/// * `reader` - Reader over the cell-based count store.
/// * `signatures` - Slice of `SignatureGenes` to calculate the scores for
/// * `cells_to_keep` - Vector of indices with the cells to keep.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A `Vec<Vec<f32>>` with cells x scores per gene set (pair)
pub fn calculate_vision_streaming<S: SingleCellReading>(
    reader: &S,
    gene_signs: &[SignatureGenes],
    cells_to_keep: &[usize],
    verbose: usize,
) -> Result<Vec<Vec<f32>>, BixverseErrors> {
    const CHUNK_SIZE: usize = 50000;

    let verbosity = parse_verbosity_level(verbose);

    let total_chunks = cells_to_keep.len().div_ceil(CHUNK_SIZE);
    let no_genes = reader.get_header().total_genes;

    let mut all_results: Vec<Vec<f32>> = Vec::with_capacity(cells_to_keep.len());

    for (chunk_idx, cell_indices_chunk) in cells_to_keep.chunks(CHUNK_SIZE).enumerate() {
        let start_chunk = Instant::now();

        let cell_chunks = reader.read_cells_parallel(cell_indices_chunk)?;

        let chunk_scores: Vec<Vec<f32>> = cell_chunks
            .par_iter()
            .map(|chunk| calculate_vision_scores_for_cell(chunk, gene_signs, no_genes))
            .collect();

        all_results.extend(chunk_scores);

        if verbosity.detailed_verbosity() {
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

    Ok(all_results)
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
/// * `knn_distances` - KNN distances (cells x k), ascending
/// * `squared_distances` - `true` when `knn_distances` already holds `d^2`.
///   Follows from the metric the neighbours came from, so derive it with
///   [`crate::single_cell::sc_processing::knn::distances_are_squared`] rather
///   than guessing: `"euclidean"` and `"l2"`
///   are pre-squared, `"cosine"` and `"manhattan"` are not.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
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
    squared_distances: bool,
    verbose: usize,
) -> (Vec<f64>, Vec<f64>) {
    let verbosity = parse_verbosity_level(verbose);

    let start = Instant::now();

    let knn_weights = calc_knn_weights(&knn_distances, squared_distances);

    if verbosity.normal_verbosity() {
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

    if verbosity.normal_verbosity() {
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

    if verbosity.normal_verbosity() {
        println!("Calculated p-values: {:.2?}", start.elapsed());
    }

    (pathway_consistency, p_vals)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Values from `VISION/R/Projections.R::computeKNNWeights` run on the same
    /// input: `sigma = rowMaxs(d)`, `exp(-1 * (d * d) / sigma^2)`, then divided
    /// by `rowSums`. The row normalisation is the part that was missing.
    #[test]
    fn test_vision_kernel_matches_the_r_reference() {
        let distances = vec![
            vec![0.5, 1.0, 2.0, 4.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ];

        let expected = [
            vec![0.320_621_29, 0.305_938_97, 0.253_632_3, 0.119_807_42],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
        ];

        let weights = calc_knn_weights(&distances, false);

        for (got, want) in weights.iter().zip(expected.iter()) {
            for (a, b) in got.iter().zip(want.iter()) {
                assert_relative_eq!(a, b, epsilon = 1e-6);
            }
            assert_relative_eq!(got.iter().sum::<f32>(), 1.0, epsilon = 1e-6);
        }
    }

    /// The `"euclidean"` metric hands back `d^2`, so the squared reading of the
    /// same neighbourhood has to reproduce the plain one.
    #[test]
    fn test_vision_kernel_handles_squared_distances() {
        let plain = vec![vec![0.5, 1.0, 2.0, 4.0]];
        let squared: Vec<Vec<f32>> = plain
            .iter()
            .map(|row| row.iter().map(|d| d * d).collect())
            .collect();

        let from_plain = calc_knn_weights(&plain, false);
        let from_squared = calc_knn_weights(&squared, true);

        for (a, b) in from_plain[0].iter().zip(from_squared[0].iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }
}
