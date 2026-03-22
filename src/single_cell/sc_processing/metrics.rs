//! Contains other metrics for single cell, for example to assess batch
//! effects, see Büttner, et al., Nat. Methods, 2019

use indexmap::IndexSet;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use statrs::distribution::ChiSquared;
use statrs::distribution::ContinuousCDF;

use crate::assert_same_len;
use crate::core::math::vector_helpers::rank_vector;
use crate::prelude::*;
use crate::single_cell::sc_processing::pca::scale_csc_chunk;
use crate::utils::simd::*;

//////////
// kBET //
//////////

/// Results from the kBET calculation
pub struct KbetResult {
    /// Per-cell p-values from the chi-square test
    pub p_values: Vec<f64>,
    /// Per-cell chi-square statistics
    pub chi_square_stats: Vec<f64>,
    /// Mean chi-square statistic (effect size measure, independent of k)
    pub mean_chi_square: f64,
    /// Median chi-square statistic (robust to outliers)
    pub median_chi_square: f64,
}

/// Calculate kBET-based mixing scores on kNN data
///
/// Uses Pearson's chi-square with Yates' continuity correction for the
/// two-batch case (DoF = 1).
///
/// ### Params
///
/// * `knn_data` - KNN data. Outer vector represents the cells, inner vector
///   the neighbour indices.
/// * `batches` - Vector indicating the batch of each cell.
///
/// ### Returns
///
/// A `KbetResult` with per-cell p-values, chi-square statistics, and summary
/// measures.
pub fn kbet(knn_data: &[Vec<usize>], batches: &[usize]) -> KbetResult {
    let mut batch_counts = FxHashMap::default();
    for &batch in batches {
        *batch_counts.entry(batch).or_insert(0usize) += 1;
    }
    let total = batches.len() as f64;
    let batch_ids: Vec<usize> = batch_counts.keys().copied().collect();
    let n_batches = batch_ids.len();
    let dof = (n_batches - 1) as f64;
    let use_yates = n_batches == 2;

    let chi_sq_dist = ChiSquared::new(dof).unwrap();

    let results: Vec<(f64, f64)> = knn_data
        .par_iter()
        .map(|neighbours| {
            let k = neighbours.len() as f64;
            let mut neighbours_count = FxHashMap::default();
            for &neighbour_idx in neighbours {
                *neighbours_count
                    .entry(batches[neighbour_idx])
                    .or_insert(0usize) += 1;
            }

            let mut chi_square = 0.0;
            for &batch_id in &batch_ids {
                let expected = k * (batch_counts[&batch_id] as f64 / total);
                let observed = *neighbours_count.get(&batch_id).unwrap_or(&0) as f64;
                let diff = if use_yates {
                    (observed - expected).abs() - 0.5
                } else {
                    observed - expected
                };
                chi_square += diff * diff / expected;
            }

            let p_value = 1.0 - chi_sq_dist.cdf(chi_square);
            (chi_square, p_value)
        })
        .collect();

    let chi_square_stats: Vec<f64> = results.iter().map(|(c, _)| *c).collect();
    let p_values: Vec<f64> = results.iter().map(|(_, p)| *p).collect();

    let mean_chi_square = chi_square_stats.iter().sum::<f64>() / chi_square_stats.len() as f64;

    let mut sorted_chi = chi_square_stats.clone();
    sorted_chi.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median_chi_square = if sorted_chi.len() % 2 == 0 {
        (sorted_chi[sorted_chi.len() / 2 - 1] + sorted_chi[sorted_chi.len() / 2]) / 2.0
    } else {
        sorted_chi[sorted_chi.len() / 2]
    };

    KbetResult {
        p_values,
        chi_square_stats,
        mean_chi_square,
        median_chi_square,
    }
}

///////////////////////////
// BatchSilhouetteScores //
///////////////////////////

/// Results from batch silhouette width calculation
pub struct BatchSilhouetteResult {
    /// Per-cell silhouette scores in [-1, 1]
    pub per_cell: Vec<f32>,
    /// Mean silhouette width (closer to 0 = better mixing)
    pub mean_asw: f32,
    /// Median silhouette width
    pub median_asw: f32,
}

/// Compute batch silhouette width from kNN data
///
/// For each cell, uses its k nearest neighbours to estimate:
///   a = mean distance to neighbours of same batch
///   b = min over other batches of mean distance to neighbours of that batch
///   s = (b - a) / max(a, b)
///
/// ### Params
///
/// * `knn_indices` - Neighbour indices per cell (N x k)
/// * `knn_distances` - Neighbour distances per cell (N x k)
/// * `batch_labels` - Batch assignment per cell (length N)
///
/// ### Returns
///
/// `BatchSilhouetteResult` with per-cell and summary scores
pub fn batch_silhouette_width_knn(
    knn_indices: &[Vec<usize>],
    knn_distances: &[Vec<f32>],
    batch_labels: &[usize],
) -> BatchSilhouetteResult {
    let n = knn_indices.len();
    let n_batches = batch_labels.iter().max().map(|&x| x + 1).unwrap_or(0);

    assert_eq!(knn_distances.len(), n);
    assert!(n_batches >= 2, "Need at least 2 batches for silhouette");

    let per_cell: Vec<f32> = knn_indices
        .par_iter()
        .zip(knn_distances.par_iter())
        .enumerate()
        .map(|(i, (indices, distances))| {
            let b_i = batch_labels[i];
            let mut batch_sum = vec![0.0f32; n_batches];
            let mut batch_count = vec![0u32; n_batches];

            for (&j, &dist) in indices.iter().zip(distances.iter()) {
                let b_j = batch_labels[j];
                batch_sum[b_j] += dist;
                batch_count[b_j] += 1;
            }

            let a = if batch_count[b_i] > 0 {
                batch_sum[b_i] / batch_count[b_i] as f32
            } else {
                0.0
            };

            let mut b = f32::INFINITY;
            for batch_idx in 0..n_batches {
                if batch_idx == b_i || batch_count[batch_idx] == 0 {
                    continue;
                }
                let mean_dist = batch_sum[batch_idx] / batch_count[batch_idx] as f32;
                if mean_dist < b {
                    b = mean_dist;
                }
            }

            if b == f32::INFINITY {
                // No neighbours from other batches at all
                return 0.0;
            }

            let max_ab = a.max(b);
            if max_ab > 0.0 { (b - a) / max_ab } else { 0.0 }
        })
        .collect();

    let mean_asw = per_cell.iter().sum::<f32>() / n as f32;

    let mut sorted = per_cell.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median_asw = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };

    BatchSilhouetteResult {
        per_cell,
        mean_asw,
        median_asw,
    }
}

//////////////////
// Pairwise cor //
//////////////////

/// Calculate the correlations between certain combinations of genes
///
/// `gene_indices_1[i]` is correlated against `gene_indices_2[i]`.
///
/// ### Params
///
/// * `f_path` - Path to the gene-based (CSC) binary file.
/// * `gene_indices_1` - First set of gene indices.
/// * `gene_indices_2` - Second set of gene indices (same length).
/// * `cells_to_keep` - Indices of cells to include.
/// * `spearman` - Use Spearman (rank-based) correlation.
///
/// ### Returns
///
/// Vector of correlations, one per pair.
pub fn pairwise_gene_correlations(
    f_path: &str,
    gene_indices_1: &[usize],
    gene_indices_2: &[usize],
    cells_to_keep: &[usize],
    spearman: bool,
) -> Vec<f32> {
    assert_same_len!(gene_indices_1, gene_indices_2);

    let n_cells = cells_to_keep.len();
    let cell_set: IndexSet<u32> = cells_to_keep.iter().map(|&x| x as u32).collect();

    // unique genes, order-preserving!!!
    let mut unique_genes: IndexSet<usize> = IndexSet::default();
    for &idx in gene_indices_1.iter().chain(gene_indices_2.iter()) {
        unique_genes.insert(idx);
    }
    let unique_vec: Vec<usize> = unique_genes.iter().copied().collect();

    // Load and filter
    let reader = ParallelSparseReader::new(f_path).unwrap();
    let mut gene_chunks = reader.read_gene_parallel(&unique_vec);

    gene_chunks.par_iter_mut().for_each(|chunk| {
        chunk.filter_selected_cells(&cell_set);
    });

    // densify, optionally rank, then standardise
    let standardised: Vec<Vec<f32>> = gene_chunks
        .par_iter()
        .map(|chunk| {
            if spearman {
                // Densify -> rank -> standardise
                let mut dense = vec![0_f32; n_cells];
                for (idx, &row_idx) in chunk.indices.iter().enumerate() {
                    dense[row_idx as usize] = chunk.data_norm[idx].to_f32();
                }
                let dense = rank_vector(&dense);
                let mean = sum_simd_f32(&dense) / n_cells as f32;
                let var = variance_simd_f32(&dense, mean) / (n_cells as f32 - 1.0);
                let std = var.sqrt();
                if std < 1e-8 {
                    vec![0_f32; n_cells]
                } else {
                    dense.iter().map(|&x| (x - mean) / std).collect()
                }
            } else {
                let (scaled, _, _) = scale_csc_chunk(chunk, n_cells);
                scaled
            }
        })
        .collect();

    // Pairwise correlations via dot product
    let denom = n_cells as f32 - 1.0;

    gene_indices_1
        .par_iter()
        .zip(gene_indices_2.par_iter())
        .map(|(&g1, &g2)| {
            let a = &standardised[unique_genes.get_index_of(&g1).unwrap()];
            let b = &standardised[unique_genes.get_index_of(&g2).unwrap()];
            let cor = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>() / denom;
            cor.clamp(-1_f32, 1_f32) // avoid floating ops instabilities
        })
        .collect()
}
