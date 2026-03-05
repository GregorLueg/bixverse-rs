//! Implementation of the miloR differential abundance approach on top of kNN
//! graphs, see Dann, et al., Nat Biotechnol, 2022

use ann_search_rs::annoy::AnnoyIndex;
use ann_search_rs::hnsw::HnswIndex;
use ann_search_rs::nndescent::NNDescent;
use ann_search_rs::utils::dist::{Dist, parse_ann_dist};
use faer::MatRef;
use rayon::prelude::*;

use crate::core::math::vector_helpers::median;
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Structure for MiloR algorithm parameters
///
/// ### Fields
///
/// **MiloR params**
///
/// * `prop` - Proportion of cells to sample as neighbourhood indices
/// * `k_refine` - Number of neighbours to use for refinement
/// * `refinement_strategy` - Strategy for refining sampled indices
///   (`"approximate"`, `"bruteforce"`, or `"index"`)
/// * `index_type` - Type of kNN index to use (`"annoy"` or `"hnsw"`)
/// * `knn_params` - The knnParams via the `KnnParams` structure.
pub struct MiloRParams {
    /// Proportion of cells to sample as neighbourhood indices
    pub prop: f64,
    /// Number of neighbours to use for refinement
    pub k_refine: usize,
    /// Strategy for refining sampled indices (`"approximate"`, `"bruteforce"`,
    /// or `"index"`)
    pub refinement_strategy: String,
    /// Type of kNN index to use (`"annoy"`, `"hnsw"` or `"nndescent"`)
    pub index_type: String,
    /// Parameters for the various approximate nearest neighbour searches
    /// in ann-search-rs
    pub knn_params: KnnParams,
}

/// Enum wrapper for different kNN index implementations
///
/// ### Variants
///
/// * `Annoy` - Approximate nearest neighbour index using trees
/// * `Hnsw` - Hierarchical navigable small world graph index
/// * `NNDescent` - Nearest neighbour descent index
pub enum KnnIndex {
    /// The Annoy index
    Annoy(AnnoyIndex<f32>),
    /// The HNSW index
    Hnsw(HnswIndex<f32>),
    /// NNDescent
    NNDescent(NNDescent<f32>),
}

impl KnnIndex {
    /// Generate a new instance of the kNN index
    ///
    /// ### Params
    ///
    /// * `embd` - The embedding matrix of cells x features to use to the
    ///   the generation
    /// * `knn_params` - The KnnParams with distance type, number of trees, etc.
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Initialised `KnnIndex`.
    pub fn new(
        embd: MatRef<f32>,
        index_type: KnnIndexType,
        knn_params: &KnnParams,
        seed: usize,
        verbose: bool,
    ) -> Self {
        match index_type {
            KnnIndexType::AnnoyIndex => {
                let dist = ann_search_rs::utils::dist::parse_ann_dist(&knn_params.ann_dist)
                    .unwrap_or_default();

                KnnIndex::Annoy(AnnoyIndex::new(embd, knn_params.n_tree, dist, seed))
            }
            KnnIndexType::HnswIndex => KnnIndex::Hnsw(HnswIndex::build(
                embd,
                knn_params.m,
                knn_params.ef_construction,
                &knn_params.ann_dist,
                seed,
                verbose,
            )),
            KnnIndexType::NNDescentIndex => {
                let dist = ann_search_rs::utils::dist::parse_ann_dist(&knn_params.ann_dist)
                    .unwrap_or_default();

                KnnIndex::NNDescent(NNDescent::new(
                    embd,
                    dist,
                    None,
                    None,
                    None,
                    None,
                    knn_params.delta,
                    knn_params.diversify_prob,
                    seed,
                    verbose,
                ))
            }
        }
    }

    /// Query for k nearest neighbours of a single point
    ///
    /// ### Params
    ///
    /// * `query_point` - The slice of values defining the query point
    /// * `knn_params` - The KnnParams with distance type, search budget, etc.
    /// * `k` - Number of neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of `(neighbour indices, distances to neighbours)`
    pub fn query_single(
        &self,
        query_point: &[f32],
        knn_params: &KnnParams,
        k: usize,
    ) -> (Vec<usize>, Vec<f32>) {
        match self {
            KnnIndex::Annoy(index) => index.query(query_point, k, knn_params.search_budget),
            KnnIndex::Hnsw(index) => index.query(query_point, k, knn_params.ef_search),
            KnnIndex::NNDescent(index) => index.query(query_point, k, None),
        }
    }
}

/// Enum specifying which kNN index type to use
///
/// ### Variants
///
/// * `AnnoyIndex` - Use Annoy index
/// * `HnswIndex` - Use HNSW index
#[allow(clippy::enum_variant_names)]
pub enum KnnIndexType {
    /// Annoy
    AnnoyIndex,
    /// HNSW
    HnswIndex,
    /// NNDescent
    NNDescentIndex,
}

//////////////
// Sampling //
//////////////

/// Enum specifying the refinement strategy for neighbourhood sampling
#[derive(Debug, Clone, Copy)]
pub enum RefinementStrategy {
    /// Search within k neighbours only
    Approximate,
    /// Linear search through all cells
    BruteForce,
    /// Use existing kNN index for search
    IndexBased,
}

/// Helper function to parse the refinement strategy
///
/// ### Params
///
/// * `s` - String specifying the strategy to use
///
/// ### Returns
///
/// The Option of the chosen `RefinementStrategy`
pub fn parse_refinement_strategy(s: &str) -> Option<RefinementStrategy> {
    match s.to_lowercase().as_str() {
        "approximate" => Some(RefinementStrategy::Approximate),
        "bruteforce" => Some(RefinementStrategy::BruteForce),
        "index" => Some(RefinementStrategy::IndexBased),
        _ => None,
    }
}

/// Helper function to parse the kNN index type
///
/// ### Params
///
/// * `s` - String specifying which kNN index to use
///
/// ### Returns
///
/// The Option of the chosen `KnnIndexType`
pub fn parse_index_type(s: &str) -> Option<KnnIndexType> {
    match s.to_lowercase().as_str() {
        "annoy" => Some(KnnIndexType::AnnoyIndex),
        "hnsw" => Some(KnnIndexType::HnswIndex),
        "nndescent" => Some(KnnIndexType::NNDescentIndex),
        _ => None,
    }
}

/// Helper function to compute the median positions
///
/// ### Params
///
/// * `embd` - The embedding matrix that was used for the generation of the kNN
///   graph.
/// * `neighbours` - Slice of indices for the neighbours
///
/// ### Returns
///
/// Vector of median features
fn compute_median_position(embd: MatRef<f32>, neighbours: &[usize]) -> Vec<f32> {
    let n_feature = embd.ncols();

    let mut median_point = vec![0.0f32; n_feature];
    for feat_idx in 0..n_feature {
        let values = neighbours
            .iter()
            .map(|&nb_idx| embd[(nb_idx, feat_idx)])
            .collect::<Vec<f32>>();

        median_point[feat_idx] = median(&values).unwrap_or(0_f32);
    }

    median_point
}

/// Find the cell nearest to a median position within a subset of candidates
///
/// ### Params
///
/// * `embd` - The embedding matrix
/// * `median_point` - The median position to query
/// * `candidates` - Indices of candidate cells to search within
/// * `metric` - Distance metric to use
///
/// ### Returns
///
/// Index of the nearest cell within the candidate subset
fn find_nearest_in_subset(
    embd: MatRef<f32>,
    median_point: &[f32],
    candidates: &[usize],
    metric: &Dist,
) -> usize {
    let median_row = MatRef::from_row_major_slice(median_point, 1, embd.ncols()).row(0);

    candidates
        .par_iter()
        .map(|&idx| {
            let dist = compute_distance_knn(median_row, embd.row(idx), metric);
            (idx, dist)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0
}

/// Find the cell nearest to a median position using brute force search
///
/// ### Params
///
/// * `embd` - The embedding matrix
/// * `median_point` - The median position to query
/// * `metric` - Distance metric to use
///
/// ### Returns
///
/// Index of the nearest cell in the entire dataset
fn find_nearest_bruteforce(embd: MatRef<f32>, median_point: &[f32], metric: &Dist) -> usize {
    let median_row = MatRef::from_row_major_slice(median_point, 1, embd.ncols()).row(0);

    (0..embd.nrows())
        .into_par_iter()
        .map(|idx| {
            let dist = compute_distance_knn(median_row, embd.row(idx), metric);
            (idx, dist)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0
}

/// Find the cell nearest to a median position using the kNN index
///
/// ### Params
///
/// * `index` - The kNN index to query
/// * `knn_params` - Parameters for the kNN search
/// * `median_point` - The median position to query
///
/// ### Returns
///
/// Index of the nearest cell
fn find_nearest_with_index(
    index: &KnnIndex,
    knn_params: &KnnParams,
    median_point: &[f32],
) -> usize {
    let (indices, _) = index.query_single(median_point, knn_params, 1);

    indices[0]
}

/// Refine neighbourhood sampling by shifting indices towards local median
/// positions
///
/// ### Params
///
/// * `embd` - The embedding matrix
/// * `knn_indices` - The kNN graph as adjacency list
/// * `sampled_indices` - Initial sampled cell indices
/// * `k_refine` - Number of neighbours to use for computing median
/// * `knn_params` - Parameters for distance calculation
/// * `strategy` - Refinement strategy to use
/// * `knn_index` - Optional kNN index for index-based strategy
/// * `verbose` - Whether to print progress messages
///
/// ### Returns
///
/// Refined cell indices after shifting to nearest median positions
#[allow(clippy::too_many_arguments)]
pub fn refine_sampling_with_strategy(
    embd: MatRef<f32>,
    knn_indices: &[Vec<usize>],
    sampled_indices: &[usize],
    k_refine: usize,
    knn_params: &KnnParams,
    strategy: &RefinementStrategy,
    knn_index: Option<&KnnIndex>,
    verbose: bool,
) -> Vec<usize> {
    if verbose {
        println!("Running refined sampling");
    }

    let mut refined = Vec::with_capacity(sampled_indices.len());

    let dist_metric = parse_ann_dist(&knn_params.ann_dist).unwrap_or_default();

    for &sample_idx in sampled_indices {
        let mut neighbours = Vec::with_capacity(k_refine);
        for j in 0..k_refine.min(knn_indices[0].len()) {
            let neighbour_idx = knn_indices[sample_idx][j];
            neighbours.push(neighbour_idx);
        }

        let median_point = compute_median_position(embd, &neighbours);

        let best_idx = match strategy {
            RefinementStrategy::Approximate => {
                find_nearest_in_subset(embd, &median_point, &neighbours, &dist_metric)
            }
            RefinementStrategy::BruteForce => {
                find_nearest_bruteforce(embd, &median_point, &dist_metric)
            }
            RefinementStrategy::IndexBased => {
                if let Some(index) = knn_index {
                    find_nearest_with_index(index, knn_params, &median_point)
                } else {
                    // Fallback to brute force
                    find_nearest_bruteforce(embd, &median_point, &dist_metric)
                }
            }
        };

        refined.push(best_idx);
    }

    refined
}

/// Compute distances to the k-th nearest neighbour for each index cell
///
/// ### Params
///
/// * `embd` - The embedding matrix
/// * `knn_indices` - The kNN graph as adjacency list
/// * `index_cells` - Indices of neighbourhood centre cells
/// * `kth_col` - Which neighbour to compute distance to (0-indexed)
///
/// ### Returns
///
/// Vector of distances to k-th neighbour for each index cell
pub fn compute_kth_distances_from_matrix(
    embd: MatRef<f32>,
    knn_indices: &[Vec<usize>],
    index_cells: &[usize],
    kth_col: usize,
) -> Vec<f64> {
    index_cells
        .par_iter()
        .map(|&cell_idx| {
            let kth_neighbour = knn_indices[cell_idx][kth_col];

            compute_distance_knn(
                embd.row(cell_idx),
                embd.row(kth_neighbour),
                &Dist::Euclidean,
            ) as f64
        })
        .collect()
}

/// Build sparse neighbourhood matrix in COO (triplet) format
///
/// Each neighbourhood includes the index cell plus its k nearest neighbours.
///
/// ### Params
///
/// * `knn_indices` - The kNN graph as adjacency list
/// * `index_cells` - Indices of neighbourhood centre cells
///
/// ### Returns
///
/// Tuple of `(row_indices, col_indices, values)` in COO format where
/// each neighbourhood is a column and non-zero entries indicate membership
pub fn build_nhood_matrix(
    knn_indices: &[Vec<usize>],
    index_cells: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let k = knn_indices[0].len();
    let n_nhoods = index_cells.len();

    // Pre-allocate (over-estimate)
    let mut row_indices = Vec::with_capacity(n_nhoods * (k + 1));
    let mut col_indices = Vec::with_capacity(n_nhoods * (k + 1));
    let mut values = Vec::with_capacity(n_nhoods * (k + 1));

    for (nh_idx, &cell_idx) in index_cells.iter().enumerate() {
        row_indices.push(cell_idx);
        col_indices.push(nh_idx);
        values.push(1.0);

        for j in 0..k {
            let neighbor_idx = knn_indices[cell_idx][j];
            row_indices.push(neighbor_idx);
            col_indices.push(nh_idx);
            values.push(1.0);
        }
    }

    (row_indices, col_indices, values)
}
