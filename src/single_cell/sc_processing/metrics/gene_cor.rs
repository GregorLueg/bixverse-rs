//! Gene-gene correlations over a single-cell count store.
//!
//! Not an integration metric, but it lives with them because it is the same
//! kind of summary statistic asked of a processed dataset.

use indexmap::IndexSet;
use rayon::prelude::*;
use std::time::Instant;

use crate::assert_same_len;
use crate::core::math::sparse::{
    SparseColMoments, sparse_col_moments, sparse_pairwise_correlations,
};
use crate::prelude::*;

//////////////////
// Pairwise cor //
//////////////////

/// Calculate the correlations between certain combinations of genes
///
/// `gene_indices_1[i]` is correlated against `gene_indices_2[i]`.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `gene_indices_1` - First set of gene indices.
/// * `gene_indices_2` - Second set of gene indices (same length).
/// * `cells_to_keep` - Indices of cells to include.
/// * `spearman` - Use Spearman (rank-based) correlation.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Vector of correlations, one per pair.
pub fn pairwise_gene_correlations<S: SingleCellReading>(
    reader: &S,
    gene_indices_1: &[usize],
    gene_indices_2: &[usize],
    cells_to_keep: &[usize],
    spearman: bool,
    verbose: usize,
) -> Result<Vec<f32>, BixverseErrors> {
    assert_same_len!(gene_indices_1, gene_indices_2);
    let verbosity = parse_verbosity_level(verbose);
    if verbosity.normal_verbosity() {
        println!("Calculating pairwise correlations between the genes of interest.")
    }
    let start = Instant::now();

    let n_cells = cells_to_keep.len();
    let cell_set: IndexSet<u32> = cells_to_keep.iter().map(|&x| x as u32).collect();

    // unique genes, order-preserving!!!
    let mut unique_genes: IndexSet<usize> = IndexSet::default();
    for &idx in gene_indices_1.iter().chain(gene_indices_2.iter()) {
        unique_genes.insert(idx);
    }
    let unique_vec: Vec<usize> = unique_genes.iter().copied().collect();

    // Load and filter
    let gene_chunks = reader.read_gene_parallel_filtered(&unique_vec, &cell_set)?;

    let end_load = start.elapsed();

    if verbosity.detailed_verbosity() {
        println!(
            " Pairwise gene correlations: Loaded in data in {:.2?}",
            end_load
        );
    }

    let start_moments = Instant::now();

    assert_same_len!(gene_chunks, unique_vec);

    // moments per gene, straight off the stored entries. No densification.
    let moments: Vec<SparseColMoments> = gene_chunks
        .par_iter()
        .map(|chunk| {
            let values: Vec<f32> = chunk.data_norm.iter().map(|v| v.to_f32()).collect();
            sparse_col_moments(&chunk.indices, &values, n_cells, spearman)
        })
        .collect();

    let end_moments = start_moments.elapsed();

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
        println!("Calculated pairwise correlations in {:.2?}", total)
    }

    Ok(res)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod pairwise_cor_tests {
    use super::*;
    use crate::core::math::vector_helpers::{pearson_correlation, rank_vector};
    use crate::single_cell::sc_data::in_memory_io::InMemorySparseReader;
    use crate::single_cell::sc_traits::F16;
    use approx::assert_relative_eq;
    use rand::prelude::*;

    /// Build a CSC `cells x genes` matrix from dense gene columns.
    ///
    /// `data` holds the same values as `data_2` cast to `u32`; nothing in this
    /// path reads the raw layer, but [`InMemorySparseReader`] needs it present
    /// to compute library sizes.
    fn csc_from_columns(columns: &[Vec<f32>]) -> CompressedSparseData2<u32, f32> {
        let n_cells = columns[0].len();
        let mut data_2: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut indptr: Vec<u32> = vec![0];
        for col in columns {
            assert_eq!(col.len(), n_cells);
            for (i, &v) in col.iter().enumerate() {
                if v != 0.0 {
                    data_2.push(v);
                    indices.push(i as u32);
                }
            }
            indptr.push(data_2.len() as u32);
        }
        let data: Vec<u32> = data_2.iter().map(|&v| (v * 100.0) as u32).collect();
        CompressedSparseData2::new_csc(
            &data,
            &indices,
            &indptr,
            Some(&data_2),
            (n_cells, columns.len()),
        )
    }

    /// Sparse log1p-like columns, matching the metacell fixture.
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

    /// The reader narrows the normalised layer to f16 on the way out, so the
    /// reference has to be built from the same quantised values or the
    /// comparison measures storage precision rather than the calculation.
    fn quantise(columns: &[Vec<f32>]) -> Vec<Vec<f32>> {
        columns
            .iter()
            .map(|col| {
                col.iter()
                    .map(|&v| {
                        if v == 0.0 {
                            0.0
                        } else {
                            F16::from_f32(v).to_f32()
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn reference_cor(a: &[f32], b: &[f32], spearman: bool) -> f64 {
        if spearman {
            pearson_correlation(&rank_vector(a), &rank_vector(b)).unwrap()
        } else {
            pearson_correlation(a, b).unwrap()
        }
    }

    /// Every pair must agree with the dense f64 reference, for both methods.
    #[test]
    fn test_pairwise_gene_cor_sc_matches_dense_reference() {
        let n_genes = 6;
        let n_cells = 300;
        let columns = synthetic_columns(n_genes, n_cells, 0.15, 42);
        let matrix = csc_from_columns(&columns);
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let quantised = quantise(&columns);
        let cells: Vec<usize> = (0..n_cells).collect();

        let mut g1: Vec<usize> = Vec::new();
        let mut g2: Vec<usize> = Vec::new();
        for a in 0..n_genes {
            for b in (a + 1)..n_genes {
                g1.push(a);
                g2.push(b);
            }
        }

        for spearman in [false, true] {
            let got = pairwise_gene_correlations(&reader, &g1, &g2, &cells, spearman, 0).unwrap();
            for (k, (&a, &b)) in g1.iter().zip(g2.iter()).enumerate() {
                let want = reference_cor(&quantised[a], &quantised[b], spearman);
                assert_relative_eq!(got[k] as f64, want, epsilon = 1e-5);
            }
        }
    }

    /// A gene against itself is 1.0.
    #[test]
    fn test_pairwise_gene_cor_sc_self_is_one() {
        let n_cells = 200;
        let columns = synthetic_columns(3, n_cells, 0.2, 7);
        let matrix = csc_from_columns(&columns);
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let cells: Vec<usize> = (0..n_cells).collect();
        let g = [0_usize, 1, 2];

        for spearman in [false, true] {
            let got = pairwise_gene_correlations(&reader, &g, &g, &cells, spearman, 0).unwrap();
            for &c in &got {
                assert_relative_eq!(c, 1.0_f32, epsilon = 1e-5);
            }
        }
    }

    /// A gene with no variance yields 0.0, not NaN and not R's NA.
    #[test]
    fn test_pairwise_gene_cor_sc_constant_gene_is_zero() {
        let n_cells = 100;
        let mut columns = synthetic_columns(2, n_cells, 0.3, 11);
        columns.push(vec![0.0_f32; n_cells]);
        let matrix = csc_from_columns(&columns);
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let cells: Vec<usize> = (0..n_cells).collect();

        for spearman in [false, true] {
            let got =
                pairwise_gene_correlations(&reader, &[0, 1], &[2, 2], &cells, spearman, 0).unwrap();
            assert_eq!(got, vec![0.0_f32, 0.0_f32]);
        }
    }

    /// A cell subset correlates the subset, not the full column.
    ///
    /// `cells_to_keep` is also deliberately unsorted here: the reader emits
    /// indices in the order the selection was given, so anything downstream
    /// that assumes ascending cell indices breaks on exactly this input.
    #[test]
    fn test_pairwise_gene_cor_sc_respects_unsorted_cell_subset() {
        let n_cells = 240;
        let columns = synthetic_columns(4, n_cells, 0.25, 99);
        let matrix = csc_from_columns(&columns);
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let quantised = quantise(&columns);

        let mut cells: Vec<usize> = (0..n_cells).step_by(2).collect();
        let mut rng = StdRng::seed_from_u64(5);
        cells.shuffle(&mut rng);

        let g1 = [0_usize, 1, 0];
        let g2 = [1_usize, 2, 3];

        for spearman in [false, true] {
            let got = pairwise_gene_correlations(&reader, &g1, &g2, &cells, spearman, 0).unwrap();
            for (k, (&a, &b)) in g1.iter().zip(g2.iter()).enumerate() {
                let sub_a: Vec<f32> = cells.iter().map(|&i| quantised[a][i]).collect();
                let sub_b: Vec<f32> = cells.iter().map(|&i| quantised[b][i]).collect();
                let want = reference_cor(&sub_a, &sub_b, spearman);
                assert_relative_eq!(got[k] as f64, want, epsilon = 1e-5);
            }
        }
    }
}
