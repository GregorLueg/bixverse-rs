//! Metrics for meta cells such as correlations between genes and others

use indexmap::IndexSet;
use rayon::prelude::*;
use std::time::Instant;

use crate::core::math::sparse::{
    SparseColMoments, sparse_col_moments, sparse_pairwise_correlations,
};
use crate::prelude::*;

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

    // moments per gene, straight off the stored entries. No densification.
    let moments: Vec<SparseColMoments> = unique_vec
        .par_iter()
        .map(|&g| {
            if g >= n_genes {
                return Err(BixverseErrors::SliceIndexOutOfBounds {
                    index: g,
                    len: n_genes,
                });
            }

            let lo = matrix.indptr[g] as usize;
            let hi = matrix.indptr[g + 1] as usize;

            Ok(sparse_col_moments(
                &matrix.indices[lo..hi],
                &data_norm[lo..hi],
                n_cells,
                spearman,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let end_moments = start.elapsed();

    if verbosity.detailed_verbosity() {
        println!(
            " Pairwise gene correlations: Reduced the genes to their moments in {:.2?}",
            end_moments
        );
    }

    let start_cor = Instant::now();

    // Safe by construction: every pair index went into `unique_genes` above.
    let pairs: Vec<(usize, usize)> = gene_indices_1
        .iter()
        .zip(gene_indices_2.iter())
        .map(|(g1, g2)| {
            (
                unique_genes.get_index_of(g1).unwrap(),
                unique_genes.get_index_of(g2).unwrap(),
            )
        })
        .collect();

    let res = sparse_pairwise_correlations(&moments, &pairs, n_cells);

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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::vector_helpers::{pearson_correlation, rank_vector};
    #[cfg(feature = "large-test")]
    use crate::utils::simd::{sum_simd_f32, sum_squared_dev_simd_f32};
    use approx::assert_relative_eq;
    use rand::prelude::*;
    use rand::rngs::StdRng;

    /// Build a CSC `cells x genes` matrix from dense gene columns.
    ///
    /// Structural zeros are dropped, so the resulting sparsity pattern is
    /// exactly the set of non-zero entries. `data_2` is a copy of `data`,
    /// which is what the R side hands over for metacells.
    fn csc_from_columns(columns: &[Vec<f32>]) -> CompressedSparseData2<f32, f32> {
        let n_cells = columns[0].len();
        let mut data: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut indptr: Vec<u32> = vec![0];
        for col in columns {
            assert_eq!(col.len(), n_cells);
            for (i, &v) in col.iter().enumerate() {
                if v != 0.0 {
                    data.push(v);
                    indices.push(i as u32);
                }
            }
            indptr.push(data.len() as u32);
        }
        let data_2 = data.clone();
        CompressedSparseData2::new_csc(
            &data,
            &indices,
            &indptr,
            Some(&data_2),
            (n_cells, columns.len()),
        )
    }

    /// The definition of what these functions compute: Pearson on the values,
    /// Spearman on the tie-corrected ranks, both accumulated in f64.
    fn reference_cor(a: &[f32], b: &[f32], spearman: bool) -> f64 {
        if spearman {
            pearson_correlation(&rank_vector(a), &rank_vector(b)).unwrap()
        } else {
            pearson_correlation(a, b).unwrap()
        }
    }

    /// Sparse log1p-like columns: mostly structural zeros, the rest positive
    /// and in the range normalised counts actually occupy.
    fn synthetic_columns(n_genes: usize, n_cells: usize, density: f64, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n_genes)
            .map(|_| {
                (0..n_cells)
                    .map(|_| {
                        if rng.random::<f64>() < density {
                            (rng.random::<f32>() * 4.0) + 0.05
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Every pair must agree with the dense f64 reference, for both methods.
    ///
    /// This is the contract the whole function exists to satisfy, so it is the
    /// test that has to survive any reimplementation of the inner loop.
    #[test]
    fn test_pairwise_gene_cor_matches_dense_reference() {
        let n_genes = 8;
        let columns = synthetic_columns(n_genes, 400, 0.15, 42);
        let matrix = csc_from_columns(&columns);

        let mut g1: Vec<usize> = Vec::new();
        let mut g2: Vec<usize> = Vec::new();
        for a in 0..n_genes {
            for b in (a + 1)..n_genes {
                g1.push(a);
                g2.push(b);
            }
        }

        for spearman in [false, true] {
            let got = pairwise_gene_correlations_in_memory(&matrix, &g1, &g2, spearman, 0).unwrap();
            for (k, (&a, &b)) in g1.iter().zip(g2.iter()).enumerate() {
                let want = reference_cor(&columns[a], &columns[b], spearman);
                assert_relative_eq!(got[k] as f64, want, epsilon = 1e-5);
            }
        }
    }

    /// A gene against itself is 1.0, which pins down the normalisation.
    #[test]
    fn test_pairwise_gene_cor_self_is_one() {
        let columns = synthetic_columns(3, 200, 0.2, 7);
        let matrix = csc_from_columns(&columns);
        let g: Vec<usize> = vec![0, 1, 2];

        for spearman in [false, true] {
            let got = pairwise_gene_correlations_in_memory(&matrix, &g, &g, spearman, 0).unwrap();
            for &c in &got {
                assert_relative_eq!(c, 1.0_f32, epsilon = 1e-5);
            }
        }
    }

    /// A gene with no variance has an undefined correlation. This crate
    /// returns 0.0 rather than R's NA, and downstream depends on that.
    #[test]
    fn test_pairwise_gene_cor_constant_gene_is_zero() {
        let mut columns = synthetic_columns(2, 100, 0.3, 11);
        columns.push(vec![0.0_f32; 100]);
        let matrix = csc_from_columns(&columns);

        for spearman in [false, true] {
            let got = pairwise_gene_correlations_in_memory(&matrix, &[0, 1], &[2, 2], spearman, 0)
                .unwrap();
            assert_eq!(got, vec![0.0_f32, 0.0_f32]);
        }
    }

    /// Genes that never co-occur still have a defined correlation: both are
    /// non-constant, and the zero blocks overlap everywhere else.
    #[test]
    fn test_pairwise_gene_cor_disjoint_patterns() {
        let n_cells = 100;
        let mut a = vec![0.0_f32; n_cells];
        let mut b = vec![0.0_f32; n_cells];
        for i in 0..20 {
            a[i] = 1.0 + i as f32 * 0.1;
            b[n_cells - 1 - i] = 1.0 + i as f32 * 0.1;
        }
        let matrix = csc_from_columns(&[a.clone(), b.clone()]);

        for spearman in [false, true] {
            let got =
                pairwise_gene_correlations_in_memory(&matrix, &[0], &[1], spearman, 0).unwrap();
            let want = reference_cor(&a, &b, spearman);
            assert_relative_eq!(got[0] as f64, want, epsilon = 1e-5);
        }
    }

    /// A CSR matrix is rejected rather than silently read along the wrong axis.
    #[test]
    fn test_pairwise_gene_cor_rejects_csr() {
        let data = [1.0_f32, 2.0];
        let indices = [0_u32, 1];
        let indptr = [0_u32, 1, 2];
        let matrix = CompressedSparseData2::new_csr(&data, &indices, &indptr, Some(&data), (2, 2));
        assert!(pairwise_gene_correlations_in_memory(&matrix, &[0], &[1], false, 0).is_err());
    }

    /// An out-of-range gene index errors rather than panicking.
    #[test]
    fn test_pairwise_gene_cor_gene_index_out_of_bounds() {
        let columns = synthetic_columns(2, 50, 0.3, 3);
        let matrix = csc_from_columns(&columns);
        assert!(pairwise_gene_correlations_in_memory(&matrix, &[0], &[9], false, 0).is_err());
    }

    /// The dense f32 formulation this function used to run.
    ///
    /// Kept only so the accuracy test below has something to compare against.
    /// Densify, standardise with f32 accumulators, then one scalar dot.
    #[cfg(feature = "large-test")]
    fn dense_f32_reference(columns: &[Vec<f32>], a: usize, b: usize) -> f32 {
        let n_cells = columns[0].len();
        let standardise = |col: &Vec<f32>| -> Vec<f32> {
            let mean = sum_simd_f32(col) / n_cells as f32;
            let var = sum_squared_dev_simd_f32(col, mean) / (n_cells as f32 - 1.0);
            let std = var.sqrt();
            col.iter().map(|&x| (x - mean) / std).collect()
        };
        let za = standardise(&columns[a]);
        let zb = standardise(&columns[b]);
        let dot = za.iter().zip(zb.iter()).map(|(x, y)| x * y).sum::<f32>();
        (dot / (n_cells as f32 - 1.0)).clamp(-1.0, 1.0)
    }

    /// The sparse form is far more accurate than the dense one, measured
    /// rather than argued. 200k cells at 5% density.
    #[test]
    #[cfg(feature = "large-test")]
    // 200_000 cells x 6 genes, 5% density, 15 pairs.
    fn test_sparse_form_beats_dense_f32_accuracy() {
        let n_genes = 6;
        let n_cells = 200_000;
        let columns = synthetic_columns(n_genes, n_cells, 0.05, 17);
        let matrix = csc_from_columns(&columns);

        let mut g1: Vec<usize> = Vec::new();
        let mut g2: Vec<usize> = Vec::new();
        for a in 0..n_genes {
            for b in (a + 1)..n_genes {
                g1.push(a);
                g2.push(b);
            }
        }

        let got = pairwise_gene_correlations_in_memory(&matrix, &g1, &g2, false, 0).unwrap();

        let mut worst_sparse = 0.0_f64;
        let mut worst_dense = 0.0_f64;
        for (k, (&a, &b)) in g1.iter().zip(g2.iter()).enumerate() {
            let exact = pearson_correlation(&columns[a], &columns[b]).unwrap();
            worst_sparse = worst_sparse.max((got[k] as f64 - exact).abs());
            worst_dense =
                worst_dense.max((dense_f32_reference(&columns, a, b) as f64 - exact).abs());
        }

        assert!(
            worst_sparse < 1e-8,
            "sparse path drifted from the f64 reference by {worst_sparse:e}"
        );
        assert!(
            worst_dense > 100.0 * worst_sparse,
            "the dense f32 path was expected to stay far worse: dense {worst_dense:e}, \
             sparse {worst_sparse:e}. If this fires the two have converged and the test \
             no longer proves the sparse form buys anything."
        );
    }

    /// Against R's `cor()` rather than against this crate's own reference.
    ///
    /// The in-crate check shares `rank_vector` with the code under test, so it
    /// cannot catch a wrong tie correction. Sparse counts put 60-70% of every
    /// column into a single zero tie group, which is exactly where average
    /// ranks are easy to get subtly wrong, so R gets the last word.
    ///
    /// Fixture from `cor(a, b, method = ...)`, R 4.x, seed 1234, n = 60.
    #[test]
    fn test_pairwise_gene_cor_matches_r_cor() {
        #[rustfmt::skip]
        let a: Vec<f32> = vec![
            3.509, 0.000, 0.000, 0.000, 0.000, 0.000, 0.217, 1.319, 0.000, 0.000,
            0.000, 0.000, 0.105, 0.000, 1.006, 0.000, 2.876, 1.282, 2.084, 0.257,
            0.000, 0.000, 2.308, 0.536, 3.621, 0.000, 0.000, 0.000, 0.000, 0.109,
            0.000, 3.182, 0.000, 0.000, 0.410, 0.000, 2.127, 1.587, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.330, 0.000,
            1.333, 0.000, 0.000, 0.000, 2.724, 0.000, 0.000, 0.000, 3.756, 0.000,
        ];
        #[rustfmt::skip]
        let b: Vec<f32> = vec![
            0.000, 3.985, 0.000, 1.348, 0.000, 0.000, 0.000, 1.975, 0.000, 1.478,
            0.000, 2.560, 3.016, 2.314, 0.000, 0.000, 3.973, 0.000, 0.000, 2.357,
            0.000, 1.806, 0.964, 0.379, 0.000, 3.451, 0.989, 0.000, 4.003, 0.000,
            2.458, 0.000, 4.045, 1.552, 0.000, 0.000, 2.271, 1.768, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 2.354, 1.780, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.949, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
        ];
        #[rustfmt::skip]
        let c: Vec<f32> = vec![
            0.584, 0.000, 0.000, 2.048, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 3.259, 0.000, 0.000, 0.000, 0.000, 1.399,
            0.000, 2.086, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.028, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 3.238, 0.000, 0.000,
        ];

        let matrix = csc_from_columns(&[a, b, c]);
        let g1 = [0_usize, 0, 1];
        let g2 = [1_usize, 2, 2];

        let spearman_r = [-0.041234370042470_f64, 0.041661907489425, 0.139779931677086];
        let pearson_r = [
            -0.059892654097613_f64,
            -0.058660661115182,
            0.094370485205331,
        ];

        let got_s = pairwise_gene_correlations_in_memory(&matrix, &g1, &g2, true, 0).unwrap();
        let got_p = pairwise_gene_correlations_in_memory(&matrix, &g1, &g2, false, 0).unwrap();

        for k in 0..3 {
            assert_relative_eq!(got_s[k] as f64, spearman_r[k], epsilon = 1e-6);
            assert_relative_eq!(got_p[k] as f64, pearson_r[k], epsilon = 1e-6);
        }
    }
}
