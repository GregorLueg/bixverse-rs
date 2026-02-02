use ann_search_rs::utils::KnnValidation;
use ann_search_rs::utils::dist::Dist;
use ann_search_rs::*;
use faer::{MatRef, RowRef};
use std::time::Instant;

use crate::core::math::sparse::coo_to_csr;
use crate::prelude::*;

//////////
// Enum //
//////////

/// Enum for the different methods
#[derive(Default)]
pub enum KnnSearch {
    /// Hierarchical Navigable Small World
    #[default]
    Hnsw,
    /// Annoy-based
    Annoy,
    /// NNDescent
    NNDescent,
    /// Exhaustive
    Exhaustive,
}

/// Helper function to get the KNN method
///
/// ### Params
///
/// * `s` - Type of KNN algorithm to use
///
/// ### Returns
///
/// Option of the HvgMethod (some not yet implemented)
pub fn parse_knn_method(s: &str) -> Option<KnnSearch> {
    match s.to_lowercase().as_str() {
        "annoy" => Some(KnnSearch::Annoy),
        "hnsw" => Some(KnnSearch::Hnsw),
        "nndescent" => Some(KnnSearch::NNDescent),
        "exhaustive" => Some(KnnSearch::Exhaustive),
        _ => None,
    }
}

////////////
// Params //
////////////

/// KnnParams
///
/// ### Fields
///
/// **General**
///
/// * `knn_method` - Which of the kNN methods to use. One of `"annoy"`, `"hnsw"`
///   or `"nndescent"`.
/// * `ann_dist` - Approximate nearest neighbour distance measure. One of
///   `"euclidean"` or `"cosine"`.
/// * `k` - Number of neighbours to search.
///
/// **Annoy**
///
/// * `n_tree` - Number of trees for the generation of the index.
/// * `search_budget` - Optional search budget. If not provided, will default
///   to `k * n_trees * 20`. Good ranges for the multipler are 2 to 20.
///
/// **NN Descent**
///
/// * `delta` - Early termination criterium.
/// * `diversify_prob` - Diversifying probability at the end of the index
///   generation.
/// * `ef_budget` - Optional query budget.
///
/// **LSH**
///
/// * `bits` - Number of bits to use.
/// * `n_tables` - Number of hash tables to use.
/// * `max_candidates` - Optional query budget.
///
/// **IVF**
///
/// * `n_centroids` - Number of centroids to use.
/// * `n_probes` - Number of centroids to probe.
#[derive(Clone, Debug)]
pub struct KnnParams {
    // general params
    pub knn_method: String,
    pub ann_dist: String,
    pub k: usize,
    // annoy params
    pub n_tree: usize,
    pub search_budget: Option<usize>,
    // nn descent params
    pub diversify_prob: f32,
    pub delta: f32,
    pub ef_budget: Option<usize>,
    // hnsw
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl KnnParams {
    /// Generate a version of this with sensible base parameters
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn new() -> Self {
        Self {
            knn_method: "hnsw".to_string(),
            ann_dist: "cosine".to_string(),
            k: 15,
            n_tree: 50,
            search_budget: None,
            diversify_prob: 0.5,
            delta: 0.001,
            ef_budget: None,
            m: 16,
            ef_construction: 200,
            ef_search: 100,
        }
    }
}

/// Default implementation for KnnParams
impl Default for KnnParams {
    fn default() -> Self {
        Self::new()
    }
}

/////////////
// Helpers //
/////////////

/// Helper function to create a kNN mat with self
///
/// ### Params
///
/// * knn_graph - The kNN graph structure in which rows represent samples and
///   the columns represent the neighbours
///
/// ### Results
///
/// Updated version with self added
pub fn build_nn_map(knn_graph: &[Vec<usize>]) -> Vec<Vec<usize>> {
    (0..knn_graph.len())
        .map(|i| {
            let mut neighbors = knn_graph[i].clone();
            neighbors.push(i);
            neighbors
        })
        .collect()
}

/// Compute distance between two points
///
/// Helper function to quickly calculate the implemented distances additionally
///
/// ### Params
///
/// * `a` - RowRef to cell a.
/// * `b` - RowRef to cell b.
///
/// ### Returns
///
/// The distance between the two cells based on the embedding.
#[inline(always)]
pub fn compute_distance_knn(a: RowRef<f32>, b: RowRef<f32>, metric: &Dist) -> f32 {
    let ncols = a.ncols();

    // fast, unsafe path for contiguous memory
    if a.col_stride() == 1 && b.col_stride() == 1 {
        unsafe {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            match metric {
                Dist::Euclidean => {
                    let mut sum = 0.0f32;
                    for i in 0..ncols {
                        let diff = *a_ptr.add(i) - *b_ptr.add(i);
                        sum += diff * diff;
                    }
                    sum.sqrt()
                }
                Dist::Cosine => {
                    let mut dot = 0.0f32;
                    let mut norm_a = 0.0f32;
                    let mut norm_b = 0.0f32;

                    for i in 0..ncols {
                        let av = *a_ptr.add(i);
                        let bv = *b_ptr.add(i);
                        dot += av * bv;
                        norm_a += av * av;
                        norm_b += bv * bv;
                    }

                    1.0 - (dot / (norm_a.sqrt() * norm_b.sqrt()))
                }
            }
        }
    } else {
        // fallback
        match metric {
            Dist::Euclidean => {
                let mut sum = 0.0f32;
                for i in 0..ncols {
                    let diff = a[i] - b[i];
                    sum += diff * diff;
                }
                sum.sqrt()
            }
            Dist::Cosine => {
                let mut dot = 0.0f32;
                let mut norm_a = 0.0f32;
                let mut norm_b = 0.0f32;

                for i in 0..ncols {
                    dot += a[i] * b[i];
                    norm_a += a[i] * a[i];
                    norm_b += b[i] * b[i];
                }

                1.0 - (dot / (norm_a.sqrt() * norm_b.sqrt()))
            }
        }
    }
}

/// Helper function to transform kNN data into CompressedSparseData
///
/// ### Params
///
/// * `knn_indices` - The indices of the k-nearest neighbours.
/// * `knn_dists` - The distances to the k-nearest neighbours.
/// * `n_obs` - Number of observations in the data.
///
/// ### Return
///
/// `CompressedSparseData` in CSR format with distances to the k-nearest
/// neighbours stored.
pub fn knn_to_sparse_dist(
    knn_indices: &[Vec<usize>],
    knn_dists: &[Vec<f32>],
    n_obs: usize,
) -> CompressedSparseData<f32> {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for i in 0..knn_indices.len() {
        for j in 0..knn_indices[i].len() {
            let neighbor = knn_indices[i][j];
            let dist = if neighbor == i { 0.0 } else { knn_dists[i][j] };

            if dist != 0.0 {
                rows.push(i);
                cols.push(neighbor);
                vals.push(dist);
            }
        }
    }

    coo_to_csr(&rows, &cols, &vals, (n_obs, n_obs))
}

////////////////////
// Main functions //
////////////////////

/// Helper function to abstract out common patterns
///
/// ### Params
///
/// * `no_neighbours` - Number of neighbours
/// * `seed` - Seed for reproducibility
/// * `verbose` - Controls verbosity of the function
/// * `build_index` - Build index function
/// * `query_index` - Query index self
/// * `validate_index` - Self validation of the data
/// * `index_name` - Name of the index
///
/// ### Returns
///
/// The kNN graph
fn build_and_query_knn<I>(
    no_neighbours: usize,
    verbose: bool,
    build_index: impl FnOnce() -> I,
    query_index: impl FnOnce(&I) -> (Vec<Vec<usize>>, Option<Vec<Vec<f32>>>),
    index_name: &str,
) -> (Vec<Vec<usize>>, I) {
    let start = Instant::now();
    let index = build_index();
    if verbose {
        println!("Generated {} index: {:.2?}", index_name, start.elapsed());
    }

    let start = Instant::now();
    let (indices, _) = query_index(&index);

    let res: Vec<Vec<usize>> = indices
        .into_iter()
        .enumerate()
        .map(|(i, mut neighbors)| {
            neighbors.retain(|&x| x != i);
            neighbors.truncate(no_neighbours);
            neighbors
        })
        .collect();

    if verbose {
        println!(
            "Identified approximate nearest neighbours via {}: {:.2?}",
            index_name,
            start.elapsed()
        );
    }

    (res, index)
}

/// Get the kNN graph based on HNSW
///
/// This function generates the kNN graph via an approximate nearest neighbour
/// search based on the HNSW algorithm (hierarchical navigable small world).
///
/// ### Params
///
/// * `mat` - Matrix in which rows represent the samples and columns the
///   respective embeddings for that sample
/// * `no_neighbours` - Number of neighbours for the KNN graph
/// * `m` - Number of connections per layer (M parameter)
/// * `ef_const` - Size of dynamic candidate list during construction
/// * `ef_search` - Size of candidate list during search (higher = better
///   recall, slower)
/// * `seed` - Seed for the HNSW algorithm
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The k-nearest neighbours based on the HNSW algorithm. Function does not
/// return self.
#[allow(clippy::too_many_arguments)]
pub fn generate_knn_hnsw(
    mat: MatRef<f32>,
    dist_metric: &str,
    no_neighbours: usize,
    m: usize,
    ef_const: usize,
    ef_search: usize,
    seed: usize,
    verbose: bool,
) -> Vec<Vec<usize>> {
    let (res, index) = build_and_query_knn(
        no_neighbours,
        verbose,
        || build_hnsw_index(mat, m, ef_const, dist_metric, seed, verbose),
        |idx| query_hnsw_self(idx, no_neighbours + 1, ef_search, false, true),
        "HNSW",
    );

    if verbose {
        let recall = index.validate_index(no_neighbours, seed, None);
        println!(
            "Recall of approximate nearest neighbours search in random subset: {:.2}",
            recall
        );
    }

    res
}

/// Get the kNN graph based on Annoy
///
/// This function generates the kNN graph based via an approximate nearest
/// neighbour search based on the Annoy algorithm (or a version thereof).
///
/// ### Params
///
/// * `mat` - Matrix in which rows represent the samples and columns the
///   respective embeddings for that sample
/// * `no_neighbours` - Number of neighbours for the KNN graph.
/// * `n_trees` - Number of trees to use for the search.
/// * `search_budget` - Optional search budget per given query. If not provided,
///   it will use `k * n_trees * 20`.
/// * `seed` - Seed for the Annoy algorithm
///
/// ### Returns
///
/// The k-nearest neighbours based on the Annoy algorithm. Function does not
/// return self.
pub fn generate_knn_annoy(
    mat: MatRef<f32>,
    dist_metric: &str,
    no_neighbours: usize,
    n_trees: usize,
    search_budget: Option<usize>,
    seed: usize,
    verbose: bool,
) -> Vec<Vec<usize>> {
    let (res, index) = build_and_query_knn(
        no_neighbours,
        verbose,
        || build_annoy_index(mat, dist_metric.to_string(), n_trees, seed),
        |idx| query_annoy_self(idx, no_neighbours + 1, search_budget, false, verbose),
        "Annoy",
    );

    if verbose {
        let recall = index.validate_index(no_neighbours, seed, None);
        println!(
            "Recall of approximate nearest neighbours search in random subset: {:.2}",
            recall
        );
    }

    res
}

/// Get the kNN graph based on NN-Descent
///
/// This function generates the kNN graph based via an approximate nearest
/// neighbour search based on the NN-Descent. The algorithm will use a
/// neighbours of neighbours logic to identify the approximate nearest
/// neighbours.
///
/// ### Params
///
/// * `mat` - Matrix in which rows represent the samples and columns the
///   respective embeddings for that sample
/// * `dist_metric` - The distance metric to use. One of `"euclidean"` or
///   `"cosine"`.
/// * `no_neighbours` - Number of neighbours for the KNN graph.
/// * `diversify_prob` - How many of the edges in the index shall be diversified
///   after index generation.
/// * `ef_budget` - Optional query search budget.
/// * `delta` - Early stop criterium for the algorithm.
/// * `seed` - Seed for the NN Descent algorithm
/// * `verbose` - Controls verbosity of the algorithm
///
/// ### Returns
///
/// The k-nearest neighbours based on the NNDescent algorithm. Function does not
/// return self.
#[allow(clippy::too_many_arguments)]
pub fn generate_knn_nndescent(
    mat: MatRef<f32>,
    dist_metric: &str,
    no_neighbours: usize,
    diversify_prob: f32,
    ef_budget: Option<usize>,
    delta: f32,
    seed: usize,
    verbose: bool,
) -> Vec<Vec<usize>> {
    let (res, index) = build_and_query_knn(
        no_neighbours,
        verbose,
        || {
            build_nndescent_index(
                mat,
                dist_metric,
                delta,
                diversify_prob,
                None,
                None,
                None,
                None,
                seed,
                verbose,
            )
        },
        |idx| query_nndescent_self(idx, no_neighbours + 1, ef_budget, false, verbose),
        "NNDescent",
    );

    if verbose {
        let recall = index.validate_index(no_neighbours, seed, None);
        println!(
            "Recall of approximate nearest neighbours search in random subset: {:.2}",
            recall
        );
    }

    res
}

/// Get the kNN graph based on an exhaustive search
///
/// ### Params
///
/// * `mat` - Matrix in which rows represent the samples and columns the
///   respective embeddings for that sample
/// * `dist_metric` - The distance metric to use. One of `"euclidean"` or
///   `"cosine"`.
/// * `no_neighbours` - Number of neighbours for the KNN graph.
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// The k-nearest neighbours based on the exhaustive linear search. Function
/// does not return self.
pub fn generate_knn_exhaustive(
    mat: MatRef<f32>,
    dist_metric: &str,
    no_neighbours: usize,
    verbose: bool,
) -> Vec<Vec<usize>> {
    let (res, _) = build_and_query_knn(
        no_neighbours,
        verbose,
        || build_exhaustive_index(mat, dist_metric),
        |idx| query_exhaustive_self(idx, no_neighbours + 1, false, verbose),
        "exhaustive linear search",
    );
    res
}

///////////////////
// With distance //
///////////////////

/// Generate the kNN indices and distances
///
/// Helper function to generate kNN indices and distances in one go
///
/// ### Params
///
/// * `embd` - The embedding matrix to use to approximate neighbours and
///   calculate distances. Cells x features.
/// * `knn_params` - The parameters for the approximate nearest neighbour
///   search.
/// * `return_dist` - Return the distances.
/// * `seed` - Seed for reproducibility
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// Tuple of `(indices of nearest neighbours, distances to these neighbours)`
pub fn generate_knn_with_dist(
    embd: MatRef<f32>,
    knn_params: &KnnParams,
    return_dist: bool,
    seed: usize,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<f32>>>) {
    // first helper function to remove self
    fn remove_self(
        mut indices: Vec<Vec<usize>>,
        distances: Option<Vec<Vec<f32>>>,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<f32>>>) {
        for idx_vec in indices.iter_mut() {
            idx_vec.remove(0);
        }
        let distances = distances.map(|mut dists| {
            for dist_vec in dists.iter_mut() {
                dist_vec.remove(0);
            }
            dists
        });
        (indices, distances)
    }

    // second helper function to time everything
    fn timed<T>(name: &str, verbose: bool, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = f();
        if verbose {
            println!("{}: {:.2?}", name, start.elapsed());
        }
        result
    }

    let knn_method = parse_knn_method(&knn_params.knn_method).unwrap_or_default();
    let k_plus_one = knn_params.k + 1;

    let (indices, distances) = match knn_method {
        KnnSearch::Annoy => {
            let index = timed("Generated Annoy index", verbose, || {
                build_annoy_index(embd, knn_params.ann_dist.clone(), knn_params.n_tree, seed)
            });
            let (indices, distances) = timed("Queried Annoy index", verbose, || {
                query_annoy_index(
                    embd,
                    &index,
                    k_plus_one,
                    knn_params.search_budget,
                    return_dist,
                    verbose,
                )
            });
            if verbose {
                let recall = index.validate_index(k_plus_one, seed, None);
                println!(
                    "Recall of approximate nearest neighbours search in random subset: {:.2}",
                    recall
                );
            }
            (indices, distances)
        }
        KnnSearch::Hnsw => {
            let index = timed("Generated HNSW index", verbose, || {
                build_hnsw_index(
                    embd,
                    knn_params.m,
                    knn_params.ef_construction,
                    &knn_params.ann_dist,
                    seed,
                    verbose,
                )
            });
            let (indices, distances) = timed("Queried HNSW index", verbose, || {
                query_hnsw_index(
                    embd,
                    &index,
                    k_plus_one,
                    knn_params.ef_search,
                    return_dist,
                    verbose,
                )
            });
            if verbose {
                let recall = index.validate_index(k_plus_one, seed, None);
                println!(
                    "Recall of approximate nearest neighbours search in random subset: {:.2}",
                    recall
                );
            }
            (indices, distances)
        }
        KnnSearch::NNDescent => {
            let index = timed("Generated NNDescent index", verbose, || {
                build_nndescent_index(
                    embd,
                    &knn_params.ann_dist,
                    knn_params.delta,
                    knn_params.diversify_prob,
                    None,
                    None,
                    None,
                    None,
                    seed,
                    verbose,
                )
            });
            let (indices, distances) = timed("Queried NNDescent index", verbose, || {
                query_nndescent_index(
                    embd,
                    &index,
                    k_plus_one,
                    knn_params.ef_budget,
                    true,
                    verbose,
                )
            });
            if verbose {
                let recall = index.validate_index(k_plus_one, seed, None);
                println!(
                    "Recall of approximate nearest neighbours search in random subset: {:.2}",
                    recall
                );
            }
            (indices, distances)
        }
        KnnSearch::Exhaustive => {
            let index = timed("Generated Exhaustive index", verbose, || {
                build_exhaustive_index(embd, &knn_params.knn_method)
            });
            timed("Queried Exhaustive index", verbose, || {
                query_exhaustive_index(embd, &index, k_plus_one, true, verbose)
            })
        }
    };

    remove_self(indices, distances)
}
