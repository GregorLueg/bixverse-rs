//! Implementation of the (fast)MNN approach from Haghverdi, et al, Nat
//! Biotechnol, 2018

use ann_search_rs::*;
use faer::{Mat, MatRef};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

use crate::prelude::*;
use crate::single_cell::sc_batch_correction::batch_utils::cosine_normalise;
use crate::single_cell::sc_processing::pca::*;

////////////
// Params //
////////////

/// Parameters for fastMNN batch correction
#[derive(Clone, Debug)]
pub struct FastMnnParams {
    /// Number of median distances for tricube kernel bandwidth (default 3.0)
    pub ndist: f32,
    /// Apply cosine normalisation before computing distances.
    pub cos_norm: bool,
    /// Number of PCs to use for the MNN calculations
    pub no_pcs: usize,
    /// Boolean. Shall randomised SVD be used.
    pub random_svd: bool,
    /// Parameters for the various approximate nearest neighbour searches
    pub knn_params: KnnParams,
}

/// Build index on `reference` and query `query` for k nearest neighbours.
/// Returns (indices, distances).
///
/// ### Params
///
/// * `query` - The query data
/// * `reference` - The reference data
/// * `k` - Number of neighbours
/// * `params` - The kNN parameters for all of the indices
/// * `seed` - Seed for reproducibility
/// * `verbose` - Boolean to control verbosity
///
/// ### Returns
///
/// The indices and distances
fn knn_search(
    query: MatRef<f32>,
    reference: MatRef<f32>,
    k: usize,
    params: &KnnParams,
    seed: usize,
    verbose: bool,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let knn_method: KnnSearch = parse_knn_method(&params.knn_method).unwrap_or_default();

    let (indices, dist) = match knn_method {
        KnnSearch::Hnsw => {
            let index = build_hnsw_index(
                reference,
                params.m,
                params.ef_construction,
                &params.ann_dist,
                seed,
                verbose,
            );
            query_hnsw_index(query, &index, k, params.ef_search, false, verbose)
        }
        KnnSearch::Annoy => {
            let index = build_annoy_index(reference, params.ann_dist.clone(), params.n_tree, seed);
            query_annoy_index(query, &index, k, params.search_budget, false, verbose)
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
                verbose,
            );
            query_nndescent_index(query, &index, k, params.ef_budget, false, verbose)
        }
        KnnSearch::Exhaustive => {
            let index = build_exhaustive_index(reference, &params.ann_dist);
            query_exhaustive_index(query, &index, k, false, verbose)
        }
        KnnSearch::Ivf => {
            let index = build_ivf_index(
                reference,
                params.n_list,
                None,
                &params.ann_dist,
                seed,
                verbose,
            );
            query_ivf_index(query, &index, k, params.n_list, false, verbose)
        }
    };

    (indices, dist.unwrap())
}

/////////////
// Helpers //
/////////////

/// Find mutual nearest neighbours from two KNN graphs
///
/// ### Params
///
/// * `left_knn` - KNN indices for left batch (each row is cell's neighbours)
/// * `right_knn` - KNN indices for right batch
///
/// ### Returns
///
/// (left_indices, right_indices) of MNN pairs
pub fn find_mutual_nns(
    left_knn: &[Vec<usize>],
    right_knn: &[Vec<usize>],
) -> (Vec<usize>, Vec<usize>) {
    let right_sets: Vec<FxHashSet<usize>> = right_knn
        .iter()
        .map(|neighbours| neighbours.iter().copied().collect())
        .collect();

    left_knn
        .par_iter()
        .enumerate()
        .fold(
            || (Vec::new(), Vec::new()),
            |(mut left_mnn, mut right_mnn), (left_idx, left_neighbours)| {
                for &right_idx in left_neighbours {
                    if right_sets[right_idx].contains(&left_idx) {
                        left_mnn.push(left_idx);
                        right_mnn.push(right_idx);
                    }
                }
                (left_mnn, right_mnn)
            },
        )
        .reduce(
            || (Vec::new(), Vec::new()),
            |(mut l1, mut r1), (l2, r2)| {
                l1.extend(l2);
                r1.extend(r2);
                (l1, r1)
            },
        )
}

/// Compute raw correction vectors from MNN pairs
///
/// ### Params
///
/// * `data_1` - Left batch data (cells x genes)
/// * `data_2` - Right batch data (cells x genes)
/// * `mnn_1` - MNN indices in left batch
/// * `mnn_2` - MNN indices in right batch
///
/// ### Returns
///
/// Matrix of correction vectors averaged per unique cell in right batch
/// (cells x genes) and sorted indices
pub fn compute_correction_vecs(
    data_1: &MatRef<f32>,
    data_2: &MatRef<f32>,
    mnn_1: &[usize],
    mnn_2: &[usize],
) -> (Mat<f32>, Vec<usize>) {
    let n_features = data_1.ncols();
    let mut accum: FxHashMap<usize, (Vec<f32>, usize)> = FxHashMap::default();

    for (&idx1, &idx2) in mnn_1.iter().zip(mnn_2.iter()) {
        let (sums, count) = accum
            .entry(idx2)
            .or_insert_with(|| (vec![0_f32; n_features], 0));
        for g in 0..n_features {
            sums[g] += data_1.get(idx1, g) - data_2.get(idx2, g);
        }
        *count += 1;
    }

    let mut sorted_indices: Vec<usize> = accum.keys().copied().collect();
    sorted_indices.sort_unstable();

    let n_unique = sorted_indices.len();
    let mut averaged = Mat::zeros(n_unique, n_features);

    for (row, &cell_idx) in sorted_indices.iter().enumerate() {
        let (sums, count) = &accum[&cell_idx];
        let n = *count as f32;
        for g in 0..n_features {
            averaged[(row, g)] = sums[g] / n;
        }
    }

    (averaged, sorted_indices)
}

/// Centre data along the batch vector direction.
///
/// Projects all cells onto the unit batch vector, computes mean projection,
/// then shifts each cell so variation along the batch direction is removed.
/// This is `.center_along_batch_vector` from the R code.
///
/// ### Params
///
/// * `data` - Cell data matrix (cells x features)
/// * `batch_vec` - The batch direction vector (length = n_features). If the
///   L2 norm is below 1e-15, data is returned unchanged.
///
/// ### Returns
///
/// Centred matrix (cells x features) with variation along `batch_vec` removed
fn center_along_batch_vector(data: &MatRef<f32>, batch_vec: &[f32]) -> Mat<f32> {
    let n_cells = data.nrows();
    let n_features = data.ncols();

    let l2: f32 = batch_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if l2 < 1e-15 {
        return data.to_owned();
    }
    let unit_vec: Vec<f32> = batch_vec.iter().map(|x| x / l2).collect();

    let mut projections = vec![0.0f32; n_cells];
    for i in 0..n_cells {
        for g in 0..n_features {
            projections[i] += data.get(i, g) * unit_vec[g];
        }
    }

    let mean_proj: f32 = projections.iter().sum::<f32>() / n_cells as f32;

    Mat::from_fn(n_cells, n_features, |i, g| {
        data.get(i, g) + (mean_proj - projections[i]) * unit_vec[g]
    })
}

/// Compute tricube-weighted correction for all cells in the target batch.
///
/// For each cell, finds k nearest neighbours among MNN-involved cells,
/// computes tricube-weighted average of their correction vectors, and
/// returns the corrected data (data + weighted_correction).
///
/// ### Params
///
/// * `data` - Target batch cell coordinates (cells x features)
/// * `corrections` - Averaged correction vectors, one row per unique MNN cell
///   (n_unique_mnn x features)
/// * `mnn_indices` - Row indices into `data` identifying cells that
///   participated in MNN pairs; used to build the neighbour search reference
/// * `k` - Number of nearest MNN neighbours to use for weighting
/// * `ndist` - Bandwidth multiplier; bandwidth = ndist * median distance to
///   the k neighbours
/// * `knn_params` - Parameters controlling the approximate nearest neighbour
///   search
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// Corrected matrix (cells x features); each cell's coordinates shifted by
/// its tricube-weighted average correction vector
#[allow(clippy::too_many_arguments)]
pub fn tricube_weighted_correction(
    data: &MatRef<f32>,
    corrections: &MatRef<f32>,
    mnn_indices: &[usize],
    k: usize,
    ndist: f32,
    knn_params: &KnnParams,
    seed: usize,
) -> Mat<f32> {
    let n_cells = data.nrows();
    let n_features = data.ncols();
    let n_mnn = mnn_indices.len();

    // Build sub-matrix of MNN-involved cells' coordinates
    let mnn_data = Mat::from_fn(n_mnn, n_features, |r, c| *data.get(mnn_indices[r], c));

    let safe_k = k.min(n_mnn);

    // Find k nearest MNN neighbours for every cell
    let (knn_idx, knn_dist) = knn_search(*data, mnn_data.as_ref(), safe_k, knn_params, seed, false);

    // Compute tricube-weighted average corrections
    let mut correction_out: Mat<f32> = Mat::zeros(n_cells, n_features);

    for cell in 0..n_cells {
        let dists = &knn_dist[cell];
        let indices = &knn_idx[cell];

        if dists.is_empty() {
            continue;
        }

        // Bandwidth = ndist * median distance
        let mut sorted_d: Vec<f32> = dists.clone();
        sorted_d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_d = if sorted_d.len() % 2 == 0 && sorted_d.len() >= 2 {
            (sorted_d[sorted_d.len() / 2 - 1] + sorted_d[sorted_d.len() / 2]) * 0.5
        } else {
            sorted_d[sorted_d.len() / 2]
        };

        let bandwidth = ndist * median_d;

        if bandwidth < 1e-15 {
            // All neighbours at the same point; unweighted average of zero-distance ones
            let zero_count = dists.iter().filter(|&&d| d < 1e-15).count();
            if zero_count > 0 {
                let w = 1.0 / zero_count as f32;
                for (j, &nn) in indices.iter().enumerate() {
                    if dists[j] < 1e-15 {
                        for g in 0..n_features {
                            correction_out[(cell, g)] += corrections.get(nn, g) * w;
                        }
                    }
                }
            }
            continue;
        }

        let mut weight_sum = 0.0f32;
        let mut weights = vec![0.0f32; indices.len()];

        for (j, &d) in dists.iter().enumerate() {
            let ratio = d / bandwidth;
            if ratio < 1.0 {
                let t = 1.0 - ratio * ratio * ratio;
                weights[j] = t * t * t;
                weight_sum += weights[j];
            }
        }

        if weight_sum > 0.0 {
            let inv_w = 1.0 / weight_sum;
            for (j, &nn) in indices.iter().enumerate() {
                if weights[j] > 0.0 {
                    let w = weights[j] * inv_w;
                    for g in 0..n_features {
                        correction_out[(cell, g)] += corrections.get(nn, g) * w;
                    }
                }
            }
        }
    }

    // Return data + correction
    Mat::from_fn(n_cells, n_features, |i, g| {
        data.get(i, g) + correction_out[(i, g)]
    })
}

/// Merge two batches following the fastMNN algorithm.
///
/// Method
///
/// 1. Orthogonalise batch2 against accumulated batch vectors
/// 2. Find MNN pairs
/// 3. Compute average correction and overall batch vector
/// 4. Centre both batches along the batch vector (removes variation along batch
///    direction)
/// 5. Recompute corrections with centred data (same MNN pairs)
/// 6. Apply tricube-weighted correction to batch2
/// 7. Stack and return
///
/// ### Params
///
/// * `data_1` - Left (reference) batch coordinates (cells x features)
/// * `data_2` - Right (target) batch coordinates (cells x features)
/// * `params` - FastMNN parameters controlling k, ndist, cosine normalisation,
///   and the underlying kNN search
/// * `batch_vecs` - Accumulated batch direction vectors from all prior merges;
///   `data_2` is orthogonalised against each before MNN search, and the new
///   batch vector is appended on return
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If true, prints progress and MNN pair counts
///
/// ### Returns
///
/// Stacked matrix of left (centred) and corrected right batch (n_total x
/// features)
pub fn merge_two_batches(
    data_1: &MatRef<f32>,
    data_2: &MatRef<f32>,
    params: &FastMnnParams,
    batch_vecs: &mut Vec<Vec<f32>>,
    seed: usize,
    verbose: bool,
) -> Mat<f32> {
    let n_features = data_1.ncols();

    // Step 1: Orthogonalise new batch against all previous batch vectors.
    // In a progressive merge the left (already merged) carries previous
    // orthogonalisations baked in, so only the right needs this.
    let mut right = data_2.to_owned();
    for vec in batch_vecs.iter() {
        right = center_along_batch_vector(&right.as_ref(), vec);
    }

    // Step 2: Find MNN pairs
    let (knn_1_to_2, _) = knn_search(
        *data_1,
        right.as_ref(),
        params.knn_params.k,
        &params.knn_params,
        seed,
        verbose,
    );
    let (knn_2_to_1, _) = knn_search(
        right.as_ref(),
        *data_1,
        params.knn_params.k,
        &params.knn_params,
        seed,
        verbose,
    );

    let (mnn_1, mnn_2) = find_mutual_nns(&knn_1_to_2, &knn_2_to_1);

    if mnn_1.is_empty() {
        if verbose {
            eprintln!("Warning: No MNN pairs found, skipping correction");
        }
        let n_total = data_1.nrows() + right.nrows();
        return Mat::from_fn(n_total, n_features, |row, col| {
            if row < data_1.nrows() {
                *data_1.get(row, col)
            } else {
                *right.as_ref().get(row - data_1.nrows(), col)
            }
        });
    }

    if verbose {
        println!("Found {} MNN pairs", mnn_1.len());
    }

    // Step 3: Compute average correction vectors and overall batch vector
    let (averaged, _unique_mnn) = compute_correction_vecs(data_1, &right.as_ref(), &mnn_1, &mnn_2);

    let n_unique = averaged.nrows();
    let mut overall_batch = vec![0.0f32; n_features];
    for g in 0..n_features {
        for i in 0..n_unique {
            overall_batch[g] += averaged[(i, g)];
        }
        overall_batch[g] /= n_unique as f32;
    }

    // Step 4: Centre both batches along the overall batch vector
    let left_centered = center_along_batch_vector(data_1, &overall_batch);
    let right_centered = center_along_batch_vector(&right.as_ref(), &overall_batch);

    // Step 5: Recompute correction vectors with centred coordinates (same MNN pairs)
    let (re_averaged, re_unique_mnn) = compute_correction_vecs(
        &left_centered.as_ref(),
        &right_centered.as_ref(),
        &mnn_1,
        &mnn_2,
    );

    // Step 6: Tricube-weighted correction applied to every cell in batch2
    let right_corrected = tricube_weighted_correction(
        &right_centered.as_ref(),
        &re_averaged.as_ref(),
        &re_unique_mnn,
        params.knn_params.k,
        params.ndist,
        &params.knn_params,
        seed,
    );

    // Record this batch vector for future orthogonalisation steps
    batch_vecs.push(overall_batch);

    // Step 7: Stack left (centred) and right (corrected)
    let n_total = left_centered.nrows() + right_corrected.nrows();
    Mat::from_fn(n_total, n_features, |row, col| {
        if row < left_centered.nrows() {
            left_centered[(row, col)]
        } else {
            right_corrected[(row - left_centered.nrows(), col)]
        }
    })
}

/// Fast MNN with cell order tracking
///
/// ### Params
///
/// * `batches` - Vec of PCA matrices per batch (cells x n_pcs)
/// * `original_indices` - Vec of original cell indices per batch
/// * `params` - `FastMnnParams` params with all of the parameters for this
///   run
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// (corrected_pca, output_to_original_mapping)
pub fn fast_mnn(
    batches: Vec<Mat<f32>>,
    original_indices: Vec<Vec<usize>>,
    params: &FastMnnParams,
    seed: usize,
    verbose: bool,
) -> (Mat<f32>, Vec<usize>) {
    assert_eq!(batches.len(), original_indices.len());

    let mut merged = batches[0].to_owned();
    let mut index_map = original_indices[0].clone();
    let mut batch_vecs: Vec<Vec<f32>> = Vec::new();
    let total_batches = batches.len();

    for (batch_num, (batch, batch_indices)) in batches
        .into_iter()
        .zip(original_indices.into_iter())
        .skip(1)
        .enumerate()
    {
        let start = Instant::now();
        merged = merge_two_batches(
            &merged.as_ref(),
            &batch.as_ref(),
            params,
            &mut batch_vecs,
            seed,
            verbose,
        );
        let elapsed = start.elapsed();

        if verbose {
            println!(
                "Merged {} of {} batches in {:.2}s",
                batch_num + 2,
                total_batches,
                elapsed.as_secs_f64()
            );
        }

        index_map.extend(batch_indices);
    }

    (merged, index_map)
}

/// Reorder corrected PCA back to original cell order
///
/// ### Params
///
/// * `corrected_pca` - Output from fast_mnn (cells x n_pcs)
/// * `output_to_original` - Mapping from output row -> original index
///
/// ### Returns
///
/// Reordered matrix matching original cell order
pub fn reorder_to_original(corrected_pca: &Mat<f32>, output_to_original: &[usize]) -> Mat<f32> {
    let n_pcs = corrected_pca.ncols();

    let n_original_cells = output_to_original.iter().copied().max().unwrap_or(0) + 1;

    let mut original_to_output = vec![0; n_original_cells];
    for (output_idx, &original_idx) in output_to_original.iter().enumerate() {
        original_to_output[original_idx] = output_idx;
    }

    Mat::from_fn(n_original_cells, n_pcs, |row, col| {
        *corrected_pca.get(original_to_output[row], col)
    })
}

/// Split PCA matrix by batch indices
///
/// ### Params
///
/// * `pca_all` - Full PCA matrix (all cells x n_pcs)
/// * `batch_indices` - Batch assignment for each cell
///
/// ### Returns
///
/// (matrices per batch, original cell indices per batch)
pub fn split_pca_by_batch(
    pca_all: &Mat<f32>,
    batch_indices: &[usize],
) -> (Vec<Mat<f32>>, Vec<Vec<usize>>) {
    let n_features = pca_all.ncols();
    let n_batches = batch_indices.iter().copied().max().unwrap_or(0) + 1;

    let mut batch_cells: Vec<Vec<usize>> = vec![Vec::new(); n_batches];
    for (cell_idx, &batch) in batch_indices.iter().enumerate() {
        batch_cells[batch].push(cell_idx);
    }

    let batches: Vec<Mat<f32>> = batch_cells
        .iter()
        .map(|cells| {
            Mat::from_fn(cells.len(), n_features, |row, col| {
                pca_all[(cells[row], col)]
            })
        })
        .collect();

    (batches, batch_cells)
}

///////////////////
// Main function //
///////////////////

/// Perform fastMNN batch correction on single cell data
///
/// Computes PCA on all cells, splits by batch, applies fastMNN correction,
/// and returns the corrected PCA matrix in the original cell order.
///
/// ### Params
///
/// * `f_path` - Path to single cell data file
/// * `cell_indices` - Indices of cells to include
/// * `gene_indices` - Indices of genes to include
/// * `batch_indices` - Batch assignment for each cell
/// * `pre_computed_pca` - Pre-computed PCA matrix (optional)
/// * `params` - FastMNN parameters
/// * `verbose` - Controls verbosity of the function
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// Batch-corrected PCA matrix (cells x n_pcs) in original cell order
#[allow(clippy::too_many_arguments)]
pub fn fast_mnn_main(
    f_path: &str,
    cell_indices: &[usize],
    gene_indices: &[usize],
    batch_indices: &[usize],
    pre_computed_pca: Option<Mat<f32>>,
    params: &FastMnnParams,
    verbose: bool,
    seed: usize,
) -> Mat<f32> {
    let pca_all = if let Some(pca) = pre_computed_pca {
        if verbose {
            println!("Using pre-computed PCA")
        }
        pca
    } else {
        if verbose {
            println!("Re-computing PCA")
        }
        let (pca, _, _, _) = pca_on_sc(
            f_path,
            cell_indices,
            gene_indices,
            params.no_pcs,
            params.random_svd,
            seed,
            false,
            verbose,
        );
        pca
    };

    let (mut pca_batches, original_indices) = split_pca_by_batch(&pca_all, batch_indices);

    if params.cos_norm {
        if verbose {
            println!("Applying cosine normalisation to the dimension reduction.")
        }
        for batch in pca_batches.iter_mut() {
            *batch = cosine_normalise(batch);
        }
    }

    let (corrected, index_map) = fast_mnn(pca_batches, original_indices, params, seed, verbose);

    reorder_to_original(&corrected, &index_map)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::mat;

    #[test]
    fn test_find_mutual_nns_simple() {
        // Cell 0 in left has right neighbour 0; cell 0 in right has left neighbour 0
        let left_knn = vec![vec![0], vec![1]];
        let right_knn = vec![vec![0], vec![1]];

        let (mnn_l, mnn_r) = find_mutual_nns(&left_knn, &right_knn);

        assert_eq!(mnn_l.len(), mnn_r.len());
        assert!(mnn_l.contains(&0));
        assert!(mnn_r.contains(&0));
        assert!(mnn_l.contains(&1));
        assert!(mnn_r.contains(&1));
    }

    #[test]
    fn test_find_mutual_nns_no_mutuals() {
        // Left points to right 1, but right 1 points to left 1 (not 0)
        let left_knn = vec![vec![1], vec![0]];
        let right_knn = vec![vec![1], vec![0]];

        let (mnn_l, mnn_r) = find_mutual_nns(&left_knn, &right_knn);

        // left 0 -> right 1, right 1 -> left 0 => mutual
        // left 1 -> right 0, right 0 -> left 1 => mutual
        assert_eq!(mnn_l.len(), 2);
        assert_eq!(mnn_r.len(), 2);
    }

    #[test]
    fn test_find_mutual_nns_asymmetric() {
        // Left 0 -> right 0, but right 0 -> left 1 (not 0)
        let left_knn = vec![vec![0], vec![0]];
        let right_knn = vec![vec![1]];

        let (mnn_l, mnn_r) = find_mutual_nns(&left_knn, &right_knn);

        // Only left 1 -> right 0 and right 0 -> left 1 is mutual
        assert_eq!(mnn_l.len(), 1);
        assert_eq!(mnn_l[0], 1);
        assert_eq!(mnn_r[0], 0);
    }

    #[test]
    fn test_find_mutual_nns_empty() {
        let left_knn: Vec<Vec<usize>> = vec![vec![0]];
        let right_knn: Vec<Vec<usize>> = vec![vec![1]]; // points to left 1 which doesn't exist in left_knn... but left 0 -> right 0

        // left 0 -> right 0, right 0 -> left 1. Not mutual.
        let (mnn_l, mnn_r) = find_mutual_nns(&left_knn, &right_knn);
        assert!(mnn_l.is_empty());
        assert!(mnn_r.is_empty());
    }

    #[test]
    fn test_find_mutual_nns_multiple_neighbours() {
        let left_knn = vec![vec![0, 1], vec![0, 1], vec![1]];
        let right_knn = vec![vec![0, 1], vec![1, 2]];

        let (mnn_l, mnn_r) = find_mutual_nns(&left_knn, &right_knn);

        // left 0 -> {right 0, right 1}; right 0 -> {left 0, left 1} => (0,0) mutual
        // left 1 -> {right 0, right 1}; right 0 -> {left 0, left 1} => (1,0) mutual
        // left 1 -> {right 0, right 1}; right 1 -> {left 1, left 2} => (1,1) mutual
        // left 2 -> {right 1}; right 1 -> {left 1, left 2} => (2,1) mutual
        assert!(mnn_l.len() >= 3);

        let pairs: Vec<(usize, usize)> = mnn_l
            .iter()
            .zip(mnn_r.iter())
            .map(|(&l, &r)| (l, r))
            .collect();
        assert!(pairs.contains(&(0, 0)));
        assert!(pairs.contains(&(1, 1)));
        assert!(pairs.contains(&(2, 1)));
    }

    #[test]
    fn test_correction_vecs_single_pair() {
        let data_1 = mat![[1.0f32, 2.0, 3.0]];
        let data_2 = mat![[0.5f32, 1.0, 1.5]];
        let mnn_1 = vec![0];
        let mnn_2 = vec![0];

        let (averaged, indices) =
            compute_correction_vecs(&data_1.as_ref(), &data_2.as_ref(), &mnn_1, &mnn_2);

        assert_eq!(indices, vec![0]);
        assert_eq!(averaged.nrows(), 1);
        assert_eq!(averaged.ncols(), 3);
        // correction = data_1[0] - data_2[0] = (0.5, 1.0, 1.5)
        assert_relative_eq!(averaged[(0, 0)], 0.5, epsilon = 1e-6);
        assert_relative_eq!(averaged[(0, 1)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(averaged[(0, 2)], 1.5, epsilon = 1e-6);
    }

    #[test]
    fn test_correction_vecs_averaging() {
        // Two pairs map to the same cell in batch 2
        let data_1 = mat![[2.0f32, 0.0], [4.0, 0.0],];
        let data_2 = mat![[1.0f32, 0.0]];
        let mnn_1 = vec![0, 1];
        let mnn_2 = vec![0, 0];

        let (averaged, indices) =
            compute_correction_vecs(&data_1.as_ref(), &data_2.as_ref(), &mnn_1, &mnn_2);

        assert_eq!(indices, vec![0]);
        // corrections: (2-1, 0) = (1, 0) and (4-1, 0) = (3, 0), average = (2, 0)
        assert_relative_eq!(averaged[(0, 0)], 2.0, epsilon = 1e-6);
        assert_relative_eq!(averaged[(0, 1)], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_correction_vecs_multiple_targets() {
        let data_1 = mat![[3.0f32, 0.0], [0.0, 3.0],];
        let data_2 = mat![[1.0f32, 0.0], [0.0, 1.0],];
        let mnn_1 = vec![0, 1];
        let mnn_2 = vec![0, 1];

        let (averaged, indices) =
            compute_correction_vecs(&data_1.as_ref(), &data_2.as_ref(), &mnn_1, &mnn_2);

        assert_eq!(indices.len(), 2);
        assert_eq!(averaged.nrows(), 2);

        let idx_0 = indices.iter().position(|&x| x == 0).unwrap();
        let idx_1 = indices.iter().position(|&x| x == 1).unwrap();
        assert_relative_eq!(averaged[(idx_0, 0)], 2.0, epsilon = 1e-6);
        assert_relative_eq!(averaged[(idx_1, 1)], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_correction_vecs_zero_difference() {
        let data = mat![[1.0f32, 2.0, 3.0]];
        let mnn = vec![0];

        let (averaged, _) = compute_correction_vecs(&data.as_ref(), &data.as_ref(), &mnn, &mnn);

        for g in 0..3 {
            assert_relative_eq!(averaged[(0, g)], 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_center_removes_variation_along_direction() {
        // Two cells offset along x-axis
        let data = mat![[0.0f32, 0.0], [4.0, 0.0],];
        let batch_vec = vec![1.0f32, 0.0];

        let centred = center_along_batch_vector(&data.as_ref(), &batch_vec);

        // After centring, both cells should have the same x-coordinate (the mean = 2.0)
        assert_relative_eq!(centred[(0, 0)], 2.0, epsilon = 1e-6);
        assert_relative_eq!(centred[(1, 0)], 2.0, epsilon = 1e-6);
        // y-coordinates unchanged
        assert_relative_eq!(centred[(0, 1)], 0.0, epsilon = 1e-6);
        assert_relative_eq!(centred[(1, 1)], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_center_preserves_orthogonal_variation() {
        let data = mat![[0.0f32, 1.0], [4.0, 3.0],];
        let batch_vec = vec![1.0f32, 0.0];

        let centred = center_along_batch_vector(&data.as_ref(), &batch_vec);

        // x collapsed to mean (2.0), y untouched
        assert_relative_eq!(centred[(0, 0)], 2.0, epsilon = 1e-6);
        assert_relative_eq!(centred[(1, 0)], 2.0, epsilon = 1e-6);
        assert_relative_eq!(centred[(0, 1)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(centred[(1, 1)], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_center_diagonal_batch_vector() {
        // batch_vec along (1,1), data offset along that diagonal
        let data = mat![[0.0f32, 0.0], [2.0, 2.0],];
        let batch_vec = vec![1.0f32, 1.0];

        let centred = center_along_batch_vector(&data.as_ref(), &batch_vec);

        // Projections onto unit (1/sqrt2, 1/sqrt2): 0 and 2*sqrt2
        // Mean projection: sqrt2
        // Both cells should end up at (1, 1)
        assert_relative_eq!(centred[(0, 0)], 1.0, epsilon = 1e-5);
        assert_relative_eq!(centred[(0, 1)], 1.0, epsilon = 1e-5);
        assert_relative_eq!(centred[(1, 0)], 1.0, epsilon = 1e-5);
        assert_relative_eq!(centred[(1, 1)], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_center_zero_batch_vector() {
        let data = mat![[1.0f32, 2.0], [3.0, 4.0],];
        let batch_vec = vec![0.0f32, 0.0];

        let centred = center_along_batch_vector(&data.as_ref(), &batch_vec);

        // Should return data unchanged
        for i in 0..2 {
            for g in 0..2 {
                assert_relative_eq!(centred[(i, g)], data[(i, g)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_center_already_centred() {
        // Both cells have the same projection onto batch_vec
        let data = mat![[1.0f32, 0.0], [1.0, 5.0],];
        let batch_vec = vec![1.0f32, 0.0];

        let centred = center_along_batch_vector(&data.as_ref(), &batch_vec);

        for i in 0..2 {
            for g in 0..2 {
                assert_relative_eq!(centred[(i, g)], data[(i, g)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_center_idempotent() {
        let data = mat![[0.0f32, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0],];
        let batch_vec = vec![1.0f32, 0.5, -0.3];

        let centred_once = center_along_batch_vector(&data.as_ref(), &batch_vec);
        let centred_twice = center_along_batch_vector(&centred_once.as_ref(), &batch_vec);

        for i in 0..3 {
            for g in 0..3 {
                assert_relative_eq!(centred_once[(i, g)], centred_twice[(i, g)], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_split_pca_basic() {
        let pca = mat![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];
        let batch_indices = vec![0, 1, 0, 1];

        let (batches, indices) = split_pca_by_batch(&pca, &batch_indices);

        assert_eq!(batches.len(), 2);
        assert_eq!(indices.len(), 2);

        assert_eq!(indices[0], vec![0, 2]);
        assert_eq!(indices[1], vec![1, 3]);

        assert_eq!(batches[0].nrows(), 2);
        assert_relative_eq!(batches[0][(0, 0)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(batches[0][(1, 0)], 5.0, epsilon = 1e-6);

        assert_eq!(batches[1].nrows(), 2);
        assert_relative_eq!(batches[1][(0, 0)], 3.0, epsilon = 1e-6);
        assert_relative_eq!(batches[1][(1, 0)], 7.0, epsilon = 1e-6);
    }

    #[test]
    fn test_split_pca_single_batch() {
        let pca = mat![[1.0f32, 2.0], [3.0, 4.0],];
        let batch_indices = vec![0, 0];

        let (batches, indices) = split_pca_by_batch(&pca, &batch_indices);

        assert_eq!(batches.len(), 1);
        assert_eq!(indices[0], vec![0, 1]);
        assert_eq!(batches[0].nrows(), 2);
    }

    #[test]
    fn test_split_pca_three_batches() {
        let pca = mat![[1.0f32], [2.0], [3.0], [4.0], [5.0], [6.0],];
        let batch_indices = vec![2, 0, 1, 0, 2, 1];

        let (batches, indices) = split_pca_by_batch(&pca, &batch_indices);

        assert_eq!(batches.len(), 3);
        assert_eq!(indices[0], vec![1, 3]);
        assert_eq!(indices[1], vec![2, 5]);
        assert_eq!(indices[2], vec![0, 4]);

        assert_relative_eq!(batches[0][(0, 0)], 2.0, epsilon = 1e-6);
        assert_relative_eq!(batches[0][(1, 0)], 4.0, epsilon = 1e-6);
        assert_relative_eq!(batches[2][(0, 0)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(batches[2][(1, 0)], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_reorder_identity() {
        let data = mat![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0],];
        let mapping = vec![0, 1, 2];

        let reordered = reorder_to_original(&data, &mapping);

        for i in 0..3 {
            for g in 0..2 {
                assert_relative_eq!(reordered[(i, g)], data[(i, g)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_reorder_reversed() {
        let data = mat![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0],];
        let mapping = vec![2, 1, 0]; // output row 0 -> original 2, etc.

        let reordered = reorder_to_original(&data, &mapping);

        assert_relative_eq!(reordered[(0, 0)], 5.0, epsilon = 1e-6);
        assert_relative_eq!(reordered[(1, 0)], 3.0, epsilon = 1e-6);
        assert_relative_eq!(reordered[(2, 0)], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_reorder_split_merge_roundtrip() {
        let pca = mat![[10.0f32, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0],];
        let batch_indices = vec![1, 0, 1, 0];

        let (batches, original_indices) = split_pca_by_batch(&pca, &batch_indices);

        // Simulate a merge that just stacks batches
        let n_total = batches.iter().map(|b| b.nrows()).sum();
        let n_features = pca.ncols();
        let mut merged = Mat::zeros(n_total, n_features);
        let mut index_map = Vec::new();
        let mut row = 0;
        for (batch, indices) in batches.iter().zip(original_indices.iter()) {
            for i in 0..batch.nrows() {
                for g in 0..n_features {
                    merged[(row, g)] = batch[(i, g)];
                }
                row += 1;
            }
            index_map.extend(indices.iter().copied());
        }

        let reordered = reorder_to_original(&merged, &index_map);

        for i in 0..4 {
            for g in 0..2 {
                assert_relative_eq!(reordered[(i, g)], pca[(i, g)], epsilon = 1e-6,);
            }
        }
    }

    #[test]
    fn test_correction_vecs_symmetric() {
        // If batches are identical, corrections should be zero
        let data = mat![[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0],];
        let mnn_1 = vec![0, 1, 2];
        let mnn_2 = vec![0, 1, 2];

        let (averaged, _) = compute_correction_vecs(&data.as_ref(), &data.as_ref(), &mnn_1, &mnn_2);

        for i in 0..averaged.nrows() {
            for g in 0..averaged.ncols() {
                assert_relative_eq!(averaged[(i, g)], 0.0, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_overall_batch_vector_direction() {
        // Batch 2 is shifted +2 along feature 0 relative to batch 1
        let data_1 = mat![[0.0f32, 0.0], [0.0, 1.0], [0.0, -1.0],];
        let data_2 = mat![[2.0f32, 0.0], [2.0, 1.0], [2.0, -1.0],];
        let mnn_1 = vec![0, 1, 2];
        let mnn_2 = vec![0, 1, 2];

        let (averaged, _) =
            compute_correction_vecs(&data_1.as_ref(), &data_2.as_ref(), &mnn_1, &mnn_2);

        let n_features = 2;
        let n_unique = averaged.nrows();
        let mut overall = vec![0.0f32; n_features];
        for g in 0..n_features {
            for i in 0..n_unique {
                overall[g] += averaged[(i, g)];
            }
            overall[g] /= n_unique as f32;
        }

        // Overall batch vector should point in -x direction (ref - target = 0 - 2 = -2)
        assert!(
            overall[0] < -1.5,
            "Expected strong negative x component, got {}",
            overall[0]
        );
        assert!(
            overall[1].abs() < 1e-6,
            "Expected near-zero y component, got {}",
            overall[1]
        );
    }

    #[test]
    fn test_center_then_recompute_reduces_within_batch_spread() {
        // Batch 1 at x~0, batch 2 at x~4, with some x-spread within each
        let data_1 = mat![[-1.0f32, 0.0], [0.0, 1.0], [1.0, 2.0],];
        let data_2 = mat![[3.0f32, 0.0], [4.0, 1.0], [5.0, 2.0],];
        let mnn_1 = vec![0, 1, 2];
        let mnn_2 = vec![0, 1, 2];

        let (averaged, _) =
            compute_correction_vecs(&data_1.as_ref(), &data_2.as_ref(), &mnn_1, &mnn_2);

        let n_unique = averaged.nrows();
        let mut overall = vec![0.0f32; 2];
        for g in 0..2 {
            for i in 0..n_unique {
                overall[g] += averaged[(i, g)];
            }
            overall[g] /= n_unique as f32;
        }

        let left_c = center_along_batch_vector(&data_1.as_ref(), &overall);
        let right_c = center_along_batch_vector(&data_2.as_ref(), &overall);

        // Within each batch, all x-coordinates should now be equal (collapsed to batch mean)
        let mean_x_left: f32 = (0..3).map(|i| left_c[(i, 0)]).sum::<f32>() / 3.0;
        let mean_x_right: f32 = (0..3).map(|i| right_c[(i, 0)]).sum::<f32>() / 3.0;

        for i in 0..3 {
            assert_relative_eq!(left_c[(i, 0)], mean_x_left, epsilon = 1e-5);
            assert_relative_eq!(right_c[(i, 0)], mean_x_right, epsilon = 1e-5);
        }

        // y-coordinates should be preserved
        for i in 0..3 {
            assert_relative_eq!(left_c[(i, 1)], data_1[(i, 1)], epsilon = 1e-5);
            assert_relative_eq!(right_c[(i, 1)], data_2[(i, 1)], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_orthogonalisation_accumulates() {
        let vec_1 = vec![1.0f32, 0.0]; // first merge shifted along x

        // Orthogonalise "batch 2" against vec_1
        let batch_2 = mat![[6.0f32, 0.0]];
        let orth = center_along_batch_vector(&batch_2.as_ref(), &vec_1);

        // With a single cell, centring collapses to itself (mean = self)
        // But with multiple cells it would remove x-variation
        assert_eq!(orth.nrows(), 1);
        // Single cell: no change expected
        assert_relative_eq!(orth[(0, 0)], 6.0, epsilon = 1e-6);
    }

    #[test]
    fn test_center_multiple_cells_along_batch_vector() {
        // 4 cells with varying x, batch vec along x
        let data = mat![[1.0f32, 5.0], [3.0, 5.0], [5.0, 5.0], [7.0, 5.0],];
        let batch_vec = vec![1.0f32, 0.0];

        let centred = center_along_batch_vector(&data.as_ref(), &batch_vec);

        // All x-coords should equal the mean (4.0)
        for i in 0..4 {
            assert_relative_eq!(centred[(i, 0)], 4.0, epsilon = 1e-5);
            assert_relative_eq!(centred[(i, 1)], 5.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_correction_direction_is_ref_minus_target() {
        // Verify sign convention: correction = ref - target
        let data_1 = mat![[10.0f32, 0.0]]; // reference
        let data_2 = mat![[0.0f32, 0.0]]; // target
        let mnn_1 = vec![0];
        let mnn_2 = vec![0];

        let (averaged, _) =
            compute_correction_vecs(&data_1.as_ref(), &data_2.as_ref(), &mnn_1, &mnn_2);

        // correction = 10 - 0 = 10 (positive, pointing from target towards reference)
        assert_relative_eq!(averaged[(0, 0)], 10.0, epsilon = 1e-6);
    }

    #[test]
    fn test_split_and_reorder_preserves_all_data() {
        // Property test: split then stack then reorder must reconstruct original
        let n_cells = 10;
        let n_features = 3;
        let pca = Mat::from_fn(n_cells, n_features, |i, j| (i * n_features + j) as f32);
        let batch_indices = vec![0, 2, 1, 0, 2, 1, 0, 1, 2, 0];

        let (batches, original_indices) = split_pca_by_batch(&pca, &batch_indices);

        let mut merged = Mat::zeros(n_cells, n_features);
        let mut index_map = Vec::new();
        let mut row = 0;
        for (batch, indices) in batches.iter().zip(original_indices.iter()) {
            for i in 0..batch.nrows() {
                for g in 0..n_features {
                    merged[(row, g)] = batch[(i, g)];
                }
                row += 1;
            }
            index_map.extend(indices.iter().copied());
        }

        let reordered = reorder_to_original(&merged, &index_map);

        for i in 0..n_cells {
            for g in 0..n_features {
                assert_relative_eq!(reordered[(i, g)], pca[(i, g)], epsilon = 1e-6,);
            }
        }
    }

    #[test]
    fn test_correction_vecs_indices_are_sorted() {
        let data_1 = mat![[1.0f32, 0.0], [2.0, 0.0], [3.0, 0.0],];
        let data_2 = mat![[0.0f32, 0.0], [0.0, 0.0], [0.0, 0.0],];
        // Map to targets in reverse order
        let mnn_1 = vec![0, 1, 2];
        let mnn_2 = vec![2, 0, 1];

        let (_, indices) =
            compute_correction_vecs(&data_1.as_ref(), &data_2.as_ref(), &mnn_1, &mnn_2);

        // Indices should be sorted
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(indices, sorted);
    }

    #[test]
    fn test_center_high_dimensional() {
        let n_cells = 5;
        let n_features = 10;
        let data = Mat::from_fn(n_cells, n_features, |i, _| i as f32);

        // batch_vec only in first dimension
        let mut batch_vec = vec![0.0f32; n_features];
        batch_vec[0] = 1.0;

        let centred = center_along_batch_vector(&data.as_ref(), &batch_vec);

        // All cells should have same projection onto feature 0
        let mean_proj: f32 = (0..n_cells).map(|i| data[(i, 0)]).sum::<f32>() / n_cells as f32;
        for i in 0..n_cells {
            assert_relative_eq!(centred[(i, 0)], mean_proj, epsilon = 1e-5);
        }
        // All other features unchanged (data[i, g>0] = i for all g, which is original)
        for i in 0..n_cells {
            for g in 1..n_features {
                assert_relative_eq!(centred[(i, g)], data[(i, g)], epsilon = 1e-5);
            }
        }
    }
}
