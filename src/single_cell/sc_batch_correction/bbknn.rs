//! Implementation of the BBKNN approach for single cell batch correction via
//! Polański, et al., Bioinformatics, 2019. Compared to some of the other
//! methods, this one does not generate a batch-corrected embedding space,
//! but a batch-corrected kNN graph for subsequent usage in clustering and
//! visualisations.

use ann_search_rs::utils::dist::{Dist, parse_ann_dist};
use ann_search_rs::*;
use extendr_api::List;
use faer::MatRef;
use rayon::prelude::*;

use crate::core::mat_struct::MatSliceView;
use crate::core::math::sparse::*;
use crate::prelude::*;

///////////
// BBKNN //
///////////

// Constants needed for the UMAP stuff
const SMOOTH_K_TOLERANCE: f32 = 1e-5;
const MIN_K_DIST_SCALE: f32 = 1e-3;

/// Structure to store the BBKNN pamareters
#[derive(Clone, Debug)]
pub struct BbknnParams {
    /// How many neighbours per batch to identify
    pub neighbours_within_batch: usize,
    /// Mixing ratio between union (1.0) and intersection (0.0).
    pub set_op_mix_ratio: f32,
    /// UMAP connectivity computation parameter, how many nearest neighbours of
    /// each cell are assumed to be fully connected.
    pub local_connectivity: f32,
    /// Trim the neighbours of each cell to these many to connectivities. May
    /// help with population independence and improve the tidiness of clustering.
    pub trim: Option<usize>,
    /// Parameters for the various approximate nearest neighbour searches
    /// in ann-search-rs
    pub knn_params: KnnParams,
}

impl BbknnParams {
    /// Generate the BbknnParams from a R list
    ///
    /// Should values not be found within the List, the parameters will default
    /// to sensible defaults.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the BBKNN parameters.
    ///
    /// ### Return
    ///
    /// The `BbknnParams` with all of the parameters.
    pub fn from_r_list(r_list: List) -> Self {
        let knn_params = KnnParams::from_r_list(r_list.clone());

        let bbknn_list = r_list.into_hashmap();

        let neighbours_within_batch = bbknn_list
            .get("neighbours_within_batch")
            .and_then(|v| v.as_integer())
            .unwrap_or(3) as usize;

        let set_op_mix_ratio = bbknn_list
            .get("set_op_mix_ratio")
            .and_then(|v| v.as_real())
            .unwrap_or(1.0) as f32;

        let local_connectivity = bbknn_list
            .get("local_connectivity")
            .and_then(|v| v.as_real())
            .unwrap_or(1.0) as f32;

        let trim = bbknn_list
            .get("trim")
            .and_then(|v| v.as_integer())
            .unwrap_or(10 * neighbours_within_batch as i32) as usize;

        Self {
            neighbours_within_batch,
            set_op_mix_ratio,
            local_connectivity,
            trim: Some(trim),
            knn_params,
        }
    }
}

///////////////////
// Helpers BBKNN //
///////////////////

/// Generate batch balanced kNN graph
///
/// The function generates on a per batch basis an approximate nearest neighbour
/// search index and then runs all cells against this batch-specific index.
/// The resulting indices across all cells/batches are combined subsequntly.
///
/// ### Params
///
/// * `mat` - The embedding matrix to use. Usually PCA. cells = rows, features
///   = columns.
/// * `batch_labels` - Slice indicating which cell belongs to which batch.
/// * `knn_method` - Which KnnSearch to use.
/// * `bbknn_params` - `BbknnParams` with the parameters for the BBKNN batch
///   correction.
/// * `seed` - Random seed.
/// * `verbose` - Controls verbosity.
///
/// ### Returns
///
/// A tuple with (nearest_neighbour_indices, nearest_neighbour_distances)
fn get_batch_balanced_knn(
    mat: MatRef<f32>,
    batch_labels: &[usize],
    knn_method: &KnnSearch,
    bbknn_params: &BbknnParams,
    seed: usize,
    verbose: bool,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let n_cells = mat.nrows();

    // get unique batches
    let unique_batches: Vec<usize> = {
        let mut batches: Vec<_> = batch_labels.to_vec();
        batches.sort_unstable();
        batches.dedup();
        batches
    };

    let dist_metric: Dist = parse_ann_dist(&bbknn_params.knn_params.ann_dist).unwrap_or_default();

    let n_batches = unique_batches.len();

    let mut all_indices = vec![vec![0; bbknn_params.neighbours_within_batch * n_batches]; n_cells];
    let mut all_distances =
        vec![vec![0.0; bbknn_params.neighbours_within_batch * n_batches]; n_cells];
    let col_indices: Vec<usize> = (0..mat.ncols()).collect();

    for (batch_idx, &batch) in unique_batches.iter().enumerate() {
        if verbose {
            println!(
                "Processing batch {} / {}: {}",
                batch_idx + 1,
                n_batches,
                batch
            );
        }

        let batch_cell_indices: Vec<usize> = batch_labels
            .iter()
            .enumerate()
            .filter(|(_, b)| **b == batch)
            .map(|(i, _)| i)
            .collect();

        let sub_matrix = MatSliceView::new(mat, &batch_cell_indices, &col_indices);
        let sub_matrix = sub_matrix.to_owned();

        let (neighbour_indices, _) = match knn_method {
            KnnSearch::Annoy => {
                // annoy path with updated functions
                let index = build_annoy_index(
                    sub_matrix.as_ref(),
                    bbknn_params.knn_params.ann_dist.clone(),
                    bbknn_params.knn_params.n_tree,
                    seed,
                );
                query_annoy_index(
                    mat,
                    &index,
                    bbknn_params.neighbours_within_batch + 1,
                    bbknn_params.knn_params.search_budget,
                    false,
                    verbose,
                )
            }
            KnnSearch::Hnsw => {
                // hnsw path with updated functions
                let index = build_hnsw_index(
                    sub_matrix.as_ref(),
                    bbknn_params.knn_params.m,
                    bbknn_params.knn_params.ef_construction,
                    &bbknn_params.knn_params.ann_dist,
                    seed,
                    verbose,
                );
                query_hnsw_index(
                    mat,
                    &index,
                    bbknn_params.neighbours_within_batch + 1,
                    bbknn_params.knn_params.ef_search,
                    true,
                    verbose,
                )
            }
            KnnSearch::NNDescent => {
                let index = build_nndescent_index(
                    sub_matrix.as_ref(),
                    &bbknn_params.knn_params.ann_dist,
                    bbknn_params.knn_params.delta,
                    bbknn_params.knn_params.diversify_prob,
                    None, // default to 30 here as the algorithm does
                    None,
                    None,
                    None,
                    seed,
                    verbose,
                );

                query_nndescent_index(
                    mat,
                    &index,
                    bbknn_params.neighbours_within_batch + 1,
                    bbknn_params.knn_params.ef_budget,
                    false,
                    verbose,
                )
            }
            &KnnSearch::Exhaustive => {
                let index =
                    build_exhaustive_index(sub_matrix.as_ref(), &bbknn_params.knn_params.ann_dist);

                query_exhaustive_index(
                    mat,
                    &index,
                    bbknn_params.neighbours_within_batch + 1,
                    false,
                    verbose,
                )
            }
        };

        let col_start = batch_idx * bbknn_params.neighbours_within_batch;

        for cell_idx in 0..n_cells {
            let mut added = 0;
            let mut k_idx = 0;

            while added < bbknn_params.neighbours_within_batch {
                let local_idx = neighbour_indices[cell_idx][k_idx];
                let global_idx = batch_cell_indices[local_idx];

                if global_idx != cell_idx {
                    let dist =
                        compute_distance_knn(mat.row(cell_idx), mat.row(global_idx), &dist_metric);

                    all_indices[cell_idx][col_start + added] = global_idx;
                    all_distances[cell_idx][col_start + added] = dist;
                    added += 1;
                }

                k_idx += 1;
            }
        }
    }

    (all_indices, all_distances)
}

/// Helper to calculate smooth kNN distances
///
/// ### Params
///
/// * `dist` - Slice of distance vectors from the kNN
/// * `k` - Number of neighbours.
/// * `local_connectivity` - Determines the minimum distance threshold (rho) by
///   interpolating between the nearest non-zero neighbors. If it's 1.5, you'd
///   interpolate between the 1st and 2nd neighbor distances.
/// * `smook_k_tol` - Tolerance parameter that controls:
///   - Whether to apply interpolation when computing rho
///   - When to stop the binary search (when |psum - target| < smook_k_tol)
/// * `min_k_dist_scale` - Minimum scaling factor for sigma to prevent it from
///   becoming too small
///
/// ### Returns
///
/// Tuple of (rho, sigma)
pub fn smooth_knn_dist(
    dist: &[Vec<f32>],
    k: f32,
    local_connectivity: f32,
    smook_k_tol: f32,
    min_k_dist_scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = dist.len();
    let n_neighbours = dist[0].len();
    let target = k.log2();

    let mean_dist = dist.iter().flat_map(|d| d.iter()).sum::<f32>() / (n * n_neighbours) as f32;

    let res: Vec<(f32, f32)> = dist
        .par_iter()
        .map(|dist_i| {
            let mut rho = 0.0_f32;

            let non_zero_dist: Vec<f32> = dist_i.iter().filter(|&&d| d > 0.0).copied().collect();

            if non_zero_dist.len() >= local_connectivity as usize {
                let index = local_connectivity.floor() as usize;
                let interpolation = local_connectivity - local_connectivity.floor();

                if index > 0 {
                    rho = non_zero_dist[index - 1];
                    if interpolation > smook_k_tol {
                        rho += interpolation * (non_zero_dist[index] - non_zero_dist[index - 1]);
                    }
                } else {
                    rho = interpolation * non_zero_dist[0];
                }
            } else if !non_zero_dist.is_empty() {
                rho = *non_zero_dist
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
            }

            // binary search for sigma
            let mut lo = 0.0_f32;
            let mut hi = f32::INFINITY;
            let mut mid = 1.0_f32;

            for _ in 0..64 {
                let mut psum = 0.0_f32;

                for j in 1..n_neighbours {
                    let d = dist_i[j] - rho;
                    if d > 0.0_f32 {
                        psum += (-d / mid).exp();
                    } else {
                        psum += 1.0;
                    }
                }

                if (psum - target).abs() < smook_k_tol {
                    break;
                }

                if psum > target {
                    hi = mid;
                    mid = (lo + hi) / 2.0;
                } else {
                    lo = mid;
                    if hi == f32::INFINITY {
                        mid *= 2.0;
                    } else {
                        mid = (lo + hi) / 2.0;
                    }
                }
            }

            let mut sigma = mid;

            if rho > 0.0 {
                let mean_i = dist_i.iter().sum::<f32>() / n_neighbours as f32;
                if sigma < min_k_dist_scale * mean_i {
                    sigma = min_k_dist_scale * mean_i;
                }
            } else if sigma < min_k_dist_scale * mean_dist {
                sigma = min_k_dist_scale * mean_dist
            }

            (sigma, rho)
        })
        .collect();

    let sigmas = res.iter().map(|r| r.0).collect();
    let rhos = res.iter().map(|r| r.1).collect();

    (sigmas, rhos)
}

/// Sorts the kNN data by distance
///
/// ### Params
///
/// * `knn_indices` - The indices of the k-nearest neighbours.
/// * `knn_dists` - The distances of the k-nearest neighbours.
///
/// ### Side effect
///
/// Resorts the indices and distances of the mutable inputs.
fn sort_knn_by_distance(knn_indices: &mut [Vec<usize>], knn_dists: &mut [Vec<f32>]) {
    for i in 0..knn_indices.len() {
        let mut pairs: Vec<_> = knn_dists[i]
            .iter()
            .zip(knn_indices[i].iter())
            .map(|(d, idx)| (*d, *idx))
            .collect();

        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for (j, (dist, idx)) in pairs.into_iter().enumerate() {
            knn_dists[i][j] = dist;
            knn_indices[i][j] = idx;
        }
    }
}

/// Compute membership strengths
///
/// Converts kNN distances into fuzzy membership weights using UMAP-style
/// exponential decay. Distances within rho get weight 1.0, distances beyond
/// decay as exp(-(dist - rho) / sigma).
///
/// ### Params
///
/// * `knn_indices` - Indices of k-nearest neighbours for each sample.
/// * `knn_dists` - Distances to k-nearest neighbours for each sample.
/// * `sigmas` - Bandwidth parameters for exponential decay per sample.
/// * `rhos` - Distance thresholds for hard-core similarity per sample.
///
/// ### Returns
///
/// Tuple of (rows, cols, vals) in COO format representing the weighted
/// connectivity graph
fn compute_membership_strengths(
    knn_indices: &[Vec<usize>],
    knn_dists: &[Vec<f32>],
    sigmas: &[f32],
    rhos: &[f32],
) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    let n_samples = knn_indices.len();
    let n_neighbours = knn_indices[0].len();

    let mut rows = Vec::with_capacity(n_samples * n_neighbours);
    let mut cols = Vec::with_capacity(n_samples * n_neighbours);
    let mut vals = Vec::with_capacity(n_samples * n_neighbours);

    for i in 0..n_samples {
        for j in 0..n_neighbours {
            let neighbor = knn_indices[i][j];

            let val = if neighbor == i {
                0.0
            } else if knn_dists[i][j] - rhos[i] <= 0.0 || sigmas[i] == 0.0 {
                1.0
            } else {
                (-(knn_dists[i][j] - rhos[i]) / sigmas[i]).exp()
            };

            rows.push(i);
            cols.push(neighbor);
            vals.push(val);
        }
    }

    (rows, cols, vals)
}

/// Apply set operations to symmetrize connectivity graph
///
/// Applies fuzzy set union/intersection operations to create a symmetrized
/// connectivity matrix. Computes:
///
/// `ratio * (A + A^T - A.*A^T) + (1-ratio) * (A.*A^T)`
///
/// where ratio = 1.0 is pure union and ratio = 0.0 is pure intersection.
///
/// ### Params
///
/// * `connectivities` - Asymmetric connectivity matrix in CSR format
/// * `set_op_mix_ratio` - Mixing ratio between union (1.0) and intersection (0.0)
///
/// ### Returns
///
/// Symmetrised connectivity matrix in CSR format
fn apply_set_operations(
    connectivities: CompressedSparseData2<f32>,
    set_op_mix_ratio: f32,
) -> CompressedSparseData2<f32> {
    let (nrow, ncol) = connectivities.shape;
    let transpose = {
        let csc_transpose = connectivities.transform(); // A^T in CSC
        // Manually reinterpret CSC of A^T as CSR
        let mut coo_rows = Vec::new();
        let mut coo_cols = Vec::new();
        let mut coo_vals = Vec::new();

        for col in 0..csc_transpose.indptr.len() - 1 {
            for idx in csc_transpose.indptr[col]..csc_transpose.indptr[col + 1] {
                let row = csc_transpose.indices[idx];
                // For A^T: swap row/col to get proper CSR representation
                coo_rows.push(col);
                coo_cols.push(row);
                coo_vals.push(csc_transpose.data[idx]);
            }
        }
        coo_to_csr(&coo_rows, &coo_cols, &coo_vals, (ncol, nrow))
    };

    // Element-wise multiply: A .* A^T
    let prod = sparse_multiply_elementwise_csr(&connectivities, &transpose);

    // set_op_mix_ratio * (A + A^T - A.*A^T) + (1 - set_op_mix_ratio) * (A.*A^T)
    let union_part = sparse_add_csr(&connectivities, &transpose);
    let union_part = sparse_subtract_csr(&union_part, &prod);
    let union_part = sparse_scalar_multiply_csr(&union_part, set_op_mix_ratio);

    let intersect_part = sparse_scalar_multiply_csr(&prod, 1.0 - set_op_mix_ratio);

    let res = sparse_add_csr(&union_part, &intersect_part);

    eliminate_zeros_csr(res)
}

/// Trim weak edges from connectivity graph
///
/// For each node, keeps only the top-k strongest connections by zeroing out
/// weights below the k-th strongest edge. Applies trimming in both row and
/// column directions to ensure symmetry.
///
/// ### Params
///
/// * `connectivities` - Connectivity matrix in CSR format
/// * `trim` - Number of strongest connections to keep per node
///
/// ### Returns
///
/// Trimmed connectivity matrix in CSR format
fn trim_graph(
    mut connectivities: CompressedSparseData2<f32>,
    trim: usize,
) -> CompressedSparseData2<f32> {
    let n = connectivities.shape.0;
    let mut thresholds = vec![0.0f32; n];

    // compute thresholds

    for i in 0..n {
        let row_start = connectivities.indptr[i];
        let row_end = connectivities.indptr[i + 1];
        let row_data = &connectivities.data[row_start..row_end];

        if row_data.len() <= trim {
            continue;
        }

        let mut sorted = row_data.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        thresholds[i] = sorted[trim - 1];
    }

    // Apply trimming twice (row then column)
    for _ in 0..2 {
        for i in 0..n {
            let row_start = connectivities.indptr[i];
            let row_end = connectivities.indptr[i + 1];

            for j in row_start..row_end {
                if connectivities.data[j] < thresholds[i] {
                    connectivities.data[j] = 0.0;
                }
            }
        }

        connectivities = eliminate_zeros_csr(connectivities);
        connectivities = connectivities.transform();
    }

    connectivities
}

////////////////
// Main BBKNN //
////////////////

/// Run batch-balanced KNN
///
/// This is the main function that implements the BBKNN logic into Rust.
///
/// ### Params
///
/// * `mat` - The embedding matrix to use. Usually PCA. cells = rows, features
///   = columns.
/// * `batch_labels` - Slice indicating which cell belongs to which batch.
/// * `bbknn_params` - `BbknnParams` with the parameters for the BBKNN batch
///   correction.
/// * `seed` - Random seed.
/// * `verbose` - Controls verbosity.
///
/// ### Returns
///
/// A tuple of two CompressedSparseData2 with `(indices, connectivities)`.
pub fn bbknn(
    mat: MatRef<f32>,
    batch_labels: &[usize],
    bbknn_params: &BbknnParams,
    seed: usize,
    verbose: bool,
) -> (CompressedSparseData2<f32>, CompressedSparseData2<f32>) {
    // parse it and worst case, I default to Annoy
    let knn_method = parse_knn_method(&bbknn_params.knn_params.knn_method).unwrap_or_default();

    if verbose {
        println!("BBKNN: generating the batch balanced kNN values.")
    }

    // 1. Get batch-balanced k-NN
    let (mut knn_indices, mut knn_dists) =
        get_batch_balanced_knn(mat, batch_labels, &knn_method, bbknn_params, seed, verbose);

    // 2. Sort the distance by KNN
    sort_knn_by_distance(&mut knn_indices, &mut knn_dists);

    if verbose {
        println!("BBKNN: Calculating UMAP-based connectivities and removing weak connections.")
    }

    // 3. Compute UMAP connectivities
    let n_neighbours = knn_indices[0].len();
    let (sigmas, rhos) = smooth_knn_dist(
        &knn_dists,
        n_neighbours as f32,
        bbknn_params.local_connectivity,
        SMOOTH_K_TOLERANCE,
        MIN_K_DIST_SCALE,
    );

    let (rows, cols, vals) = compute_membership_strengths(&knn_indices, &knn_dists, &sigmas, &rhos);
    let n_obs = mat.nrows();

    let mut connectivities = coo_to_csr(&rows, &cols, &vals, (n_obs, n_obs));

    // 4. Apply set operations
    connectivities = apply_set_operations(connectivities, bbknn_params.set_op_mix_ratio);

    if verbose {
        println!("BBKNN: Finalising data.")
    }

    // 5. Create the distance matrix
    let dist = knn_to_sparse_dist(&knn_indices, &knn_dists, n_obs);

    // 6. Trimming
    if let Some(trim_val) = bbknn_params.trim
        && trim_val > 0
    {
        connectivities = trim_graph(connectivities, trim_val);
    }

    (dist, connectivities)
}
