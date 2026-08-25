//! Contains other metrics for single cell, for example to assess batch
//! effects, see Büttner, et al., Nat. Methods, 2019

use ann_search_rs::utils::dist::euclidean_distance_static;
use faer::MatRef;
use indexmap::IndexSet;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Instant;
use thousands::*;

use crate::assert_same_len;
use crate::core::math::sparse::{
    SparseColMoments, sparse_col_moments, sparse_pairwise_correlations,
};
use crate::prelude::*;

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
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// A `KbetResult` with per-cell p-values, chi-square statistics, and summary
/// measures.
pub fn kbet(
    knn_data: &[Vec<usize>],
    batches: &[usize],
    verbose: bool,
) -> Result<KbetResult, BixverseErrors> {
    let mut batch_counts = FxHashMap::default();
    for &batch in batches {
        *batch_counts.entry(batch).or_insert(0usize) += 1;
    }
    let total = batches.len() as f64;
    let batch_ids: Vec<usize> = batch_counts.keys().copied().collect();
    let n_batches = batch_ids.len();

    if n_batches == 1 {
        return Err(BixverseErrors::NeedAtLeastTwoBatches { n_batches });
    }

    let dof = (n_batches - 1) as f64;
    let use_yates = n_batches == 2;
    let n = knn_data.len();

    if verbose {
        println!("Running kBET on {} samples", n.separate_with_underscores())
    }

    let chi_sq_dist = ChiSquared::new(dof).unwrap();
    let counter = Arc::new(AtomicUsize::new(0));

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

            if verbose {
                let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                if count.is_multiple_of(100_000) {
                    println!(
                        " kBET: processed {} / {} cells.",
                        count.separate_with_underscores(),
                        n.separate_with_underscores()
                    );
                }
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
    let median_chi_square = if sorted_chi.len().is_multiple_of(2) {
        (sorted_chi[sorted_chi.len() / 2 - 1] + sorted_chi[sorted_chi.len() / 2]) / 2.0
    } else {
        sorted_chi[sorted_chi.len() / 2]
    };

    Ok(KbetResult {
        p_values,
        chi_square_stats,
        mean_chi_square,
        median_chi_square,
    })
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

/// Compute batch average silhouette width on an embedding
///
/// For each cell, computes:
///   a = mean distance to cells of same batch
///   b = mean distance to cells of nearest other batch
///   s = (b - a) / max(a, b)
///
/// Values near 0 indicate good mixing, near 1 indicates separation.
///
/// ### Params
///
/// * `embedding` - Low-dimensional embedding (N x d)
/// * `batch_labels` - Batch assignment per cell (length N)
/// * `subsample` - Optional max cells to use. If Some and N exceeds this,
///   a random subsample is taken.
/// * `seed` - Random seed for subsampling
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// `BatchSilhouetteResult` with per-cell and summary scores
pub fn batch_silhouette_width(
    embedding: MatRef<f32>,
    batch_labels: &[usize],
    subsample: Option<usize>,
    seed: usize,
    verbose: bool,
) -> Result<BatchSilhouetteResult, BixverseErrors> {
    let n = embedding.nrows();
    let d = embedding.ncols();
    assert_eq!(batch_labels.len(), n);

    let indices: Vec<usize> = if let Some(max_n) = subsample {
        if n > max_n {
            let mut rng = StdRng::seed_from_u64(seed as u64);
            let mut idx: Vec<usize> = (0..n).collect();
            idx.shuffle(&mut rng);
            idx.truncate(max_n);
            idx.sort_unstable();
            idx
        } else {
            (0..n).collect()
        }
    } else {
        (0..n).collect()
    };

    let n_sub = indices.len();
    let sub_labels: Vec<usize> = indices.iter().map(|&i| batch_labels[i]).collect();
    let n_batches = sub_labels.iter().max().map(|&x| x + 1).unwrap_or(0);

    if n_batches == 1 {
        return Err(BixverseErrors::NeedAtLeastTwoBatches { n_batches });
    }

    if verbose {
        println!(
            "Running batch silhouette calculations on {} samples.",
            n_sub.separate_with_underscores()
        )
    }

    // pre-extract rows as contiguous slices for SIMD
    let rows: Vec<Vec<f32>> = indices
        .iter()
        .map(|&i| (0..d).map(|j| embedding[(i, j)]).collect())
        .collect();

    let counter = Arc::new(AtomicUsize::new(0));

    let per_cell: Vec<f32> = (0..n_sub)
        .into_par_iter()
        .map(|ii| {
            let b_i = sub_labels[ii];
            let mut batch_sum = vec![0.0f32; n_batches];
            let mut batch_count = vec![0u32; n_batches];

            for jj in 0..n_sub {
                if ii == jj {
                    continue;
                }
                let dist = euclidean_distance_static(&rows[ii], &rows[jj]).sqrt();
                batch_sum[sub_labels[jj]] += dist;
                batch_count[sub_labels[jj]] += 1;
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

            if verbose {
                let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                if count.is_multiple_of(100_000) {
                    println!(
                        " Batch silhouette calculations: processed {} / {} cells.",
                        count.separate_with_underscores(),
                        n_sub.separate_with_underscores()
                    );
                }
            }

            let max_ab = a.max(b);
            if max_ab > 0.0 { (b - a) / max_ab } else { 0.0 }
        })
        .collect();

    let mean_asw = per_cell.iter().sum::<f32>() / n_sub as f32;

    let mut sorted = per_cell.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median_asw = if n_sub.is_multiple_of(2) {
        (sorted[n_sub / 2 - 1] + sorted[n_sub / 2]) / 2.0
    } else {
        sorted[n_sub / 2]
    };

    Ok(BatchSilhouetteResult {
        per_cell,
        mean_asw,
        median_asw,
    })
}

//////////
// Lisi //
//////////

/// Results from the LISI calculation
pub struct LisiResult {
    /// Per-cell LISI scores (range: [1, n_batches])
    pub per_cell: Vec<f32>,
    /// Mean LISI across all cells
    pub mean_lisi: f32,
    /// Median LISI across all cells
    pub median_lisi: f32,
}

/// Compute Local Inverse Simpson's Index on batch labels
///
/// For each cell, computes the effective number of batches in its
/// neighbourhood:
///
///   LISI = 1 / sum(p_b^2)
///
/// where p_b is the proportion of neighbours belonging to batch b.
///
/// ### Params
///
/// * `knn_indices` - Neighbour indices per cell
/// * `batch_labels` - Batch assignment per cell (length N)
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// `LisiResult` with per-cell scores and summaries
pub fn batch_lisi(
    knn_indices: &[Vec<usize>],
    batch_labels: &[usize],
    verbose: bool,
) -> Result<LisiResult, BixverseErrors> {
    let n = knn_indices.len();
    let n_batches = batch_labels.iter().max().map(|&x| x + 1).unwrap_or(0);

    if n_batches == 1 {
        return Err(BixverseErrors::NeedAtLeastTwoBatches { n_batches });
    }

    if verbose {
        println!(
            "Running LISI calculations on {} samples.",
            n.separate_with_underscores()
        )
    }

    let counter = Arc::new(AtomicUsize::new(0));

    let per_cell: Vec<f32> = knn_indices
        .par_iter()
        .map(|neighbours| {
            let k = neighbours.len() as f32;
            let mut counts = vec![0u32; n_batches];

            for &j in neighbours {
                counts[batch_labels[j]] += 1;
            }

            let simpson: f32 = counts
                .iter()
                .map(|&c| {
                    let p = c as f32 / k;
                    p * p
                })
                .sum();

            if verbose {
                let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                if count.is_multiple_of(100_000) {
                    println!(
                        " LISI calculations: processed {} / {} cells.",
                        count.separate_with_underscores(),
                        n.separate_with_underscores()
                    );
                }
            }

            1.0 / simpson
        })
        .collect();

    let mean_lisi = per_cell.iter().sum::<f32>() / n as f32;

    let mut sorted = per_cell.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median_lisi = if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };

    Ok(LisiResult {
        per_cell,
        mean_lisi,
        median_lisi,
    })
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
