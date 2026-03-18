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

/// Calculate kBET-based mixing scores on kNN data
///
/// ### Params
///
/// * `knn_data` - KNN data. Outer vector represents the cells, while the inner
///   vector represents
/// * `batches` - Vector indicating the batches.
///
/// ### Return
///
/// Numerical vector indicating with the p-values from the ChiSquare test
pub fn kbet(knn_data: &Vec<Vec<usize>>, batches: &Vec<usize>) -> Vec<f64> {
    let mut batch_counts = FxHashMap::default();
    for &batch in batches {
        *batch_counts.entry(batch).or_insert(0) += 1;
    }
    let total = batches.len() as f64;
    let batch_ids: Vec<usize> = batch_counts.keys().copied().collect();
    let dof = (batch_ids.len() - 1) as f64;

    knn_data
        .par_iter()
        .map(|neighbours| {
            let k = neighbours.len() as f64;
            let mut neighbours_count = FxHashMap::default();
            for &neighbour_idx in neighbours {
                *neighbours_count.entry(batches[neighbour_idx]).or_insert(0) += 1;
            }

            // Chi-square test: Σ (observed - expected)² / expected
            let mut chi_square = 0.0;
            for &batch_id in &batch_ids {
                let expected = k * (batch_counts[&batch_id] as f64 / total);
                let observed = *neighbours_count.get(&batch_id).unwrap_or(&0) as f64;
                chi_square += (observed - expected).powi(2) / expected;
            }

            // Compute p-value from chi-square distribution
            1.0 - ChiSquared::new(dof).unwrap().cdf(chi_square)
        })
        .collect()
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
