//! Utilies for batch correction methods that are shared

use ann_search_rs::*;
use faer::{Mat, MatRef};
use rayon::prelude::*;

use crate::prelude::*;

/// Apply cosine normalisation (L2 normalisation) to each row
///
/// Normalizes each row (cell) to unit L2 norm. Rows with near-zero norm
/// are left as zeros to avoid division by zero.
///
/// ### Params
///
/// * `mat` - The matrix on which to apply the Cosine normalisation per row
///
/// ### Returns
///
/// Per row L2-normalised data
pub fn cosine_normalise(mat: &Mat<f32>) -> Mat<f32> {
    let nrows = mat.nrows();
    let ncols = mat.ncols();

    // compute norms once per row (in parallel)
    let norms: Vec<f32> = (0..nrows)
        .into_par_iter()
        .map(|row| mat.get(row, ..).norm_l2())
        .collect();

    // create normalised matrix
    Mat::from_fn(nrows, ncols, |row, col| {
        let norm = norms[row];
        if norm > 1e-8 {
            mat[(row, col)] / norm
        } else {
            0.0 // Zero-norm row stays zero
        }
    })
}

/// Process the batch labels
///
/// Deduplicates the batch labels and returns the number of unique batches
///
/// ### Params
///
/// * `batch_labels` - The batch labels (length is n_cells)
///
/// ### Returns
///
/// A tuple of `(unique_batch_labels, n_batch_labels)`
pub fn process_batch_labels(batch_labels: &[usize]) -> (Vec<usize>, usize) {
    let unique_batches: Vec<usize> = {
        let mut batches: Vec<_> = batch_labels.to_vec();
        batches.sort_unstable();
        batches.dedup();
        batches
    };

    let n_batches = unique_batches.len();

    (unique_batches, n_batches)
}

/// Standardise a features x cells matrix per column: each cell has mean 0
/// and standard deviation 1 across features. Matches Seurat's `Standardize`
/// C++ helper used inside `RunCCA` prior to the cross-covariance step.
///
/// ### Params
///
/// * `mat` - Input matrix, `features x cells`. Not modified.
///
/// ### Returns
///
/// New matrix with the same shape, per-column standardised. Columns whose
/// standard deviation is below `1e-15` are returned as zeros (matches
/// Seurat's guard against constant columns).
pub fn standardise_per_column(mat: MatRef<f32>) -> Mat<f32> {
    let n_features = mat.nrows();
    let n_cells = mat.ncols();

    let stats: Vec<(f32, f32)> = (0..n_cells)
        .into_par_iter()
        .map(|col| {
            let mut sum = 0.0_f32;
            for row in 0..n_features {
                sum += *mat.get(row, col);
            }
            let mean = sum / n_features as f32;
            let mut ss = 0.0_f32;
            for row in 0..n_features {
                let d = *mat.get(row, col) - mean;
                ss += d * d;
            }
            let sd = (ss / (n_features as f32 - 1.0).max(1.0)).sqrt();
            (mean, sd)
        })
        .collect();

    Mat::from_fn(n_features, n_cells, |row, col| {
        let (mean, sd) = stats[col];
        if sd > 1e-15 {
            (*mat.get(row, col) - mean) / sd
        } else {
            0.0
        }
    })
}

/// Build an index on `reference` and query `query` for the k nearest
/// neighbours. Dispatches over the [KnnSearch] backend chosen through
/// [KnnParams].
///
/// ### Params
///
/// * `query` - Query cells x features
/// * `reference` - Reference cells x features
/// * `k` - Number of neighbours
/// * `params` - kNN backend parameters
/// * `seed` - Random seed
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// `(indices, distances)` where `indices[i]` are the k nearest reference
/// neighbours of query cell `i`. The distances are true distances:
/// [`to_true_distances`] has already run.
pub fn batch_knn_search(
    query: MatRef<f32>,
    reference: MatRef<f32>,
    k: usize,
    params: &KnnParams,
    seed: usize,
    verbose: usize,
) -> ScKnnResults {
    let verbosity = parse_verbosity_level(verbose);
    let knn_method: KnnSearch = parse_knn_method(&params.knn_method).unwrap_or_default();

    let (indices, dist) = match knn_method {
        KnnSearch::Hnsw => {
            let index = build_hnsw_index(
                reference,
                params.m,
                params.ef_construction,
                &params.ann_dist,
                seed,
                verbosity.detailed_verbosity(),
            );
            query_hnsw_index(
                query,
                &index,
                k,
                params.ef_search,
                true,
                verbosity.detailed_verbosity(),
            )?
        }
        KnnSearch::Annoy => {
            let index = build_annoy_index(reference, &params.ann_dist, params.n_tree, seed)?;
            query_annoy_index(
                query,
                &index,
                k,
                params.search_budget,
                true,
                verbosity.detailed_verbosity(),
            )?
        }
        KnnSearch::NNDescent => {
            let index = build_nndescent_index(
                reference,
                &params.ann_dist,
                params.delta,
                params.diversify_prob,
                None,
                None,
                None,
                None,
                seed,
                verbosity.detailed_verbosity(),
            )?;
            query_nndescent_index(
                query,
                &index,
                k,
                params.ef_budget,
                true,
                verbosity.detailed_verbosity(),
            )?
        }
        KnnSearch::Exhaustive => {
            let index = build_exhaustive_index(reference, &params.ann_dist);
            query_exhaustive_index(query, &index, k, true, verbosity.detailed_verbosity())?
        }
        KnnSearch::Ivf => {
            let index = build_ivf_index(
                reference,
                params.n_list,
                None,
                &params.ann_dist,
                seed,
                verbosity.detailed_verbosity(),
            )?;
            query_ivf_index(
                query,
                &index,
                k,
                params.n_list,
                true,
                verbosity.detailed_verbosity(),
            )?
        }
        KnnSearch::KmKnn => {
            let index = build_kmknn_index(
                reference,
                &params.ann_dist,
                params.n_list,
                None,
                seed,
                verbosity.detailed_verbosity(),
            )?;
            query_kmknn_index(query, &index, k, true, verbosity.detailed_verbosity())?
        }
    };

    let mut dist = dist.expect("distances were requested from the index");
    to_true_distances(&mut dist, &params.ann_dist);

    Ok((indices, dist))
}

///////////
// Tests //
//////////

#[cfg(test)]
mod tests_batch_utils {
    use super::*;
    use approx::assert_relative_eq;

    /// Every row comes out with unit norm and its direction unchanged.
    #[test]
    fn test_cosine_normalise_basic() {
        // Regular vectors with different magnitudes
        let mat = Mat::from_fn(3, 3, |i, j| {
            match i {
                0 => [3.0, 4.0, 0.0][j], // norm = 5
                1 => [1.0, 0.0, 0.0][j], // norm = 1
                2 => [0.5, 0.5, 0.5][j], // norm = sqrt(0.75)
                _ => unreachable!(),
            }
        });

        let normalised = cosine_normalise(&mat);

        // Check all rows have unit norm
        for row in 0..3 {
            let norm: f32 = (0..3)
                .map(|col| normalised[(row, col)].powi(2))
                .sum::<f32>()
                .sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
        }

        // Check specific values
        assert_relative_eq!(normalised[(0, 0)], 3.0 / 5.0, epsilon = 1e-6);
        assert_relative_eq!(normalised[(0, 1)], 4.0 / 5.0, epsilon = 1e-6);
        assert_relative_eq!(normalised[(1, 0)], 1.0, epsilon = 1e-6);
    }

    /// The near-zero cutoff is strict, so a row sitting exactly on it is zeroed.
    #[test]
    fn test_cosine_normalise_near_zero_threshold() {
        // Test values right at the threshold
        let mat = Mat::from_fn(3, 3, |i, j| {
            match i {
                0 => [1e-9, 1e-9, 1e-9][j], // Below threshold (norm ≈ 1.7e-9)
                1 => [1e-7, 1e-7, 1e-7][j], // Above threshold (norm ≈ 1.7e-7)
                2 => [1e-8, 0.0, 0.0][j],   // Exactly at threshold (norm = 1e-8)
                _ => unreachable!(),
            }
        });

        let normalized = cosine_normalise(&mat);

        // Row 0: below threshold, should be zero
        for col in 0..3 {
            assert_eq!(normalized[(0, col)], 0.0);
        }

        // Row 1: above threshold, should be normalized
        let norm: f32 = (0..3)
            .map(|col| normalized[(1, col)].powi(2))
            .sum::<f32>()
            .sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-4);

        // Row 2: exactly at threshold (1e-8 > 1e-8 is false), should be zero
        for col in 0..3 {
            assert_eq!(normalized[(2, col)], 0.0);
        }
    }

    /// With one column the result is the sign of each entry, and zero stays zero.
    #[test]
    fn test_cosine_normalise_single_column() {
        // Edge case: single column (scalars)
        let mat = Mat::from_fn(3, 1, |i, _| match i {
            0 => 5.0,
            1 => 0.0,
            2 => -3.0,
            _ => unreachable!(),
        });

        let normalised = cosine_normalise(&mat);

        assert_relative_eq!(normalised[(0, 0)], 1.0, epsilon = 1e-6);
        assert_eq!(normalised[(1, 0)], 0.0);
        assert_relative_eq!(normalised[(2, 0)], -1.0, epsilon = 1e-6);
    }

    /// Normal, zero, large and tiny rows all normalise correctly in one call.
    #[test]
    fn test_cosine_normalise_mixed_cases() {
        // Mix of normal, zero, and edge cases
        let mat = Mat::from_fn(5, 4, |i, j| {
            match i {
                0 => [3.0, 4.0, 0.0, 0.0][j],       // Normal
                1 => [0.0, 0.0, 0.0, 0.0][j],       // Zero
                2 => [1.0, 1.0, 1.0, 1.0][j],       // Equal values
                3 => [100.0, 200.0, 300.0, 0.0][j], // Large
                4 => [0.001, 0.002, 0.002, 0.0][j], // Small
                _ => unreachable!(),
            }
        });

        let normalised = cosine_normalise(&mat);

        // Check zero row
        for col in 0..4 {
            assert_eq!(normalised[(1, col)], 0.0);
        }

        // Check all non-zero rows have unit norm
        for row in [0, 2, 3, 4] {
            let norm: f32 = (0..4)
                .map(|col| normalised[(row, col)].powi(2))
                .sum::<f32>()
                .sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
        }
    }
}
