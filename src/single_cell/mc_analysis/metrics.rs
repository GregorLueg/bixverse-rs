//! Metrics for meta cells such as correlations between genes and others

use indexmap::IndexSet;
use rayon::prelude::*;
use std::time::Instant;

use crate::core::math::vector_helpers::rank_vector;
use crate::prelude::*;
use crate::utils::simd::*;

/// Calculate the correlations between certain combinations of genes, operating
/// on an in-memory CSC `cells × genes` matrix.
///
/// `gene_indices_1[i]` is correlated against `gene_indices_2[i]`. Correlation
/// is computed on the normalised layer (`data_2`).
///
/// ### Params
///
/// * `matrix` - In-memory sparse matrix, CSC, cells × genes, with the norm
///   counts in `data_2`.
/// * `gene_indices_1` - First set of gene (column) indices.
/// * `gene_indices_2` - Second set of gene (column) indices (same length).
/// * `spearman` - Use Spearman (rank-based) correlation.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Vector of correlations, one per pair.
pub fn pairwise_gene_correlations_in_memory<T: BixverseNumeric>(
    matrix: &CompressedSparseData2<T, f32>,
    gene_indices_1: &[usize],
    gene_indices_2: &[usize],
    spearman: bool,
    verbose: usize,
) -> Result<Vec<f32>, BixverseErrors> {
    assert_same_len!(gene_indices_1, gene_indices_2);

    if !matrix.cs_type.is_csc() {
        return Err(BixverseErrors::SparseMatrixMustBeCsc);
    }

    let data_norm = matrix
        .data_2
        .as_ref()
        .ok_or(BixverseErrors::Data2NotAvailable)?;

    let start = Instant::now();
    let verbosity = parse_verbosity_level(verbose);
    if verbosity.normal_verbosity() {
        println!("Calculating pairwise correlations between the genes of interest for meta cells.")
    }

    let n_cells = matrix.shape.0;
    let n_genes = matrix.shape.1;

    // unique genes, order-preserving (same contract as the disk version)
    let mut unique_genes: IndexSet<usize> = IndexSet::default();
    for &idx in gene_indices_1.iter().chain(gene_indices_2.iter()) {
        unique_genes.insert(idx);
    }
    let unique_vec: Vec<usize> = unique_genes.iter().copied().collect();

    // densify, optionally rank, then standardise (one dense column per gene)
    let standardised: Vec<Vec<f32>> = unique_vec
        .par_iter()
        .map(|&g| {
            if g >= n_genes {
                return Err(BixverseErrors::SliceIndexOutOfBounds {
                    index: g,
                    len: n_genes,
                });
            }

            let start = matrix.indptr[g] as usize;
            let end = matrix.indptr[g + 1] as usize;

            let mut dense = vec![0_f32; n_cells];
            for p in start..end {
                dense[matrix.indices[p] as usize] = data_norm[p];
            }

            let dense = if spearman { rank_vector(&dense) } else { dense };

            let mean = sum_simd_f32(&dense) / n_cells as f32;
            let var = sum_squared_dev_simd_f32(&dense, mean) / (n_cells as f32 - 1.0);
            let std = var.sqrt();

            Ok(if std < 1e-8 {
                vec![0_f32; n_cells]
            } else {
                dense.iter().map(|&x| (x - mean) / std).collect()
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let end_densify = start.elapsed();

    if verbosity.detailed_verbosity() {
        println!(
            " Pairwise gene correlations: Densified, normalised and optionally ranked the data in {:.2?}",
            end_densify
        );
    }

    let start_cor = Instant::now();

    // pairwise correlations via dot product
    let denom = n_cells as f32 - 1.0;
    let res = gene_indices_1
        .par_iter()
        .zip(gene_indices_2.par_iter())
        .map(|(&g1, &g2)| {
            let a = &standardised[unique_genes.get_index_of(&g1).unwrap()];
            let b = &standardised[unique_genes.get_index_of(&g2).unwrap()];
            let cor = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>() / denom;
            cor.clamp(-1_f32, 1_f32)
        })
        .collect();

    let end_cor = start_cor.elapsed();
    if verbosity.detailed_verbosity() {
        println!(
            " Pairwise gene correlations: Calculated correlation coefficients in {:.2?}",
            end_cor
        );
    }

    let total = start.elapsed();
    if verbosity.normal_verbosity() {
        println!(
            "Calculated pairwise correlations for meta cells in {:.2?}",
            total
        )
    }

    Ok(res)
}
