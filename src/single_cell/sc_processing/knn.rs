//! Contains the single cell-related kNN functions. Wrappers to generate
//! the kNN graphs (with and without distances).

use ann_search_rs::utils::KnnValidation;
use ann_search_rs::utils::dist::{Dist, SimdDistance, parse_ann_dist};
use ann_search_rs::*;
use faer::{MatRef, RowRef};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::time::Instant;

use crate::core::math::sparse::coo_to_csr;
use crate::prelude::*;

///////////
// Types //
///////////

/// Single cell Knn results with optional distances
pub type ScKnnOptionVersion = Result<(Vec<Vec<usize>>, Option<Vec<Vec<f32>>>), BixverseErrors>;

/// Single cell Knn results with distances
pub type ScKnnResults = Result<(Vec<Vec<usize>>, Vec<Vec<f32>>), BixverseErrors>;

//////////
// Enum //
//////////

/// Enum for the different methods
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KnnSearch {
    #[default]
    /// K-means kNN -> fast, exhaustive one (default)
    KmKnn,
    /// Hierarchical Navigable Small World
    Hnsw,
    /// Annoy-based
    Annoy,
    /// NNDescent
    NNDescent,
    /// IVF
    Ivf,
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
        "kmknn" => Some(KnnSearch::KmKnn),
        "annoy" => Some(KnnSearch::Annoy),
        "hnsw" => Some(KnnSearch::Hnsw),
        "nndescent" => Some(KnnSearch::NNDescent),
        "exhaustive" => Some(KnnSearch::Exhaustive),
        "ivf" => Some(KnnSearch::Ivf),
        _ => None,
    }
}

////////////
// Params //
////////////

/// KnnParams
///
/// Contains the parameters for the kNN searches used in the single cell parts
/// of this crate
#[derive(Clone, Debug)]
pub struct KnnParams {
    ///  Which of the kNN methods to use. One of `"annoy"`, `"hnsw"`, `"ivf"`,
    /// `"kmknn"`, `"exhaustive"` or `"nndescent"` are supported for now.
    pub knn_method: String,
    /// Distance metric to use. One of `"euclidean"` or `"cosine"`.
    pub ann_dist: String,
    /// Number of neighbours to return
    pub k: usize,
    /// Annoy: Number of trees to build
    pub n_tree: usize,
    /// Annoy: optional search budget. If not provided, will default to k * 20
    /// per tree.
    pub search_budget: Option<usize>,
    /// NNDescent: diversification probability after generation of the graph.
    pub diversify_prob: f32,
    /// NNDescent: convergence criterium. If less than these percentage of
    /// neighbours have been udpated, the algorithm counts as converged.
    pub delta: f32,
    /// NNDescent: optional beam search budget for querying.
    pub ef_budget: Option<usize>,
    /// NNDescent: hand back the graph the descent already built instead of
    /// beam searching it. No search runs at all, so it is much cheaper, but it
    /// is only valid for a self-kNN and rows can come back shorter than `k`
    /// where the descent never filled them. Ignored by every other backend.
    pub extract_knn: bool,
    /// HNSW: connections per given layer to use
    pub m: usize,
    /// HNSW: construction budget
    pub ef_construction: usize,
    /// HNSW: search budget
    pub ef_search: usize,
    /// IVF and KmKnn: number of lists/clusters. If not provided will default to
    /// `sqrt(n)`
    pub n_list: Option<usize>,
    /// IVF: number of lists/clusters to probe. If not provided will default to
    /// `sqrt(n_list)`
    pub n_probe: Option<usize>,
}

impl KnnParams {
    /// Generate a version of this with sensible base parameters
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn new() -> Self {
        Self {
            // general
            knn_method: "kmknn".to_string(),
            ann_dist: "euclidean".to_string(),
            // annoy
            k: 15,
            n_tree: 50,
            search_budget: None,
            // nndescent
            diversify_prob: 0.0,
            delta: 0.001,
            ef_budget: None,
            extract_knn: false,
            // hnsw
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            // ivf
            n_list: None,
            n_probe: None,
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

/// Warn when `extract_knn` was asked for on a backend that cannot honour it.
///
/// Only NN-Descent leaves behind a graph that can be handed back without a
/// search, so every other backend falls through to its query path. Staying
/// silent would hide a parameter that did nothing.
///
/// ### Params
///
/// * `extract_knn` - Whether direct extraction was requested
/// * `method` - The resolved kNN backend
pub(crate) fn warn_unsupported_extract(extract_knn: bool, method: KnnSearch) {
    if extract_knn && method != KnnSearch::NNDescent {
        println!(
            "[WARNING!] 'extract_knn' is only supported by NNDescent. Ignoring it for the {:?} backend.",
            method
        );
    }
}

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

    if a.col_stride() == 1 && b.col_stride() == 1 {
        let a_slice = unsafe { std::slice::from_raw_parts(a.as_ptr(), ncols) };
        let b_slice = unsafe { std::slice::from_raw_parts(b.as_ptr(), ncols) };

        match metric {
            Dist::SquaredEuclidean => f32::euclidean_simd(a_slice, b_slice).sqrt(),
            Dist::Cosine => {
                let dot = f32::dot_simd(a_slice, b_slice);
                let norm_a = f32::calculate_l2_norm(a_slice);
                let norm_b = f32::calculate_l2_norm(b_slice);
                1.0 - (dot / (norm_a * norm_b))
            }
            Dist::Manhattan => unreachable!(),
        }
    } else {
        match metric {
            Dist::SquaredEuclidean => {
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
            Dist::Manhattan => unreachable!(),
        }
    }
}

/// Helper function to transform kNN data into CompressedSparseData2
///
/// ### Params
///
/// * `knn_indices` - The indices of the k-nearest neighbours.
/// * `knn_dists` - The distances to the k-nearest neighbours.
/// * `n_obs` - Number of observations in the data.
///
/// ### Return
///
/// `CompressedSparseData2` in CSR format with distances to the k-nearest
/// neighbours stored.
pub fn knn_to_sparse_dist(
    knn_indices: &[Vec<usize>],
    knn_dists: &[Vec<f32>],
    n_obs: usize,
) -> CompressedSparseData2<f32> {
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

    coo_to_csr(
        &rows.index_cast(),
        &cols.index_cast(),
        &vals,
        (n_obs, n_obs),
    )
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
fn build_and_query_knn<I, E>(
    no_neighbours: usize,
    verbose: bool,
    build_index: impl FnOnce() -> Result<I, E>,
    query_index: impl FnOnce(&I) -> Result<(Vec<Vec<usize>>, Option<Vec<Vec<f32>>>), E>,
    index_name: &str,
) -> Result<(Vec<Vec<usize>>, I), E> {
    let start = Instant::now();
    let index = build_index()?;
    if verbose {
        println!("Generated {} index: {:.2?}", index_name, start.elapsed());
    }

    let start = Instant::now();
    let (indices, _) = query_index(&index)?;

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

    Ok((res, index))
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
/// * `dist_metric` - Distance metric to use. One of `"euclidean"` or
///   `"cosine"`.
/// * `no_neighbours` - Number of neighbours for the KNN graph
/// * `m` - Number of connections per layer (M parameter)
/// * `ef_const` - Size of dynamic candidate list during construction
/// * `ef_search` - Size of candidate list during search (higher = better
///   recall, slower)
/// * `seed` - Seed for the HNSW algorithm
/// * `validate_index` - Shall the index be validated with an exhaustive search.
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
    validate_index: bool,
    verbose: bool,
) -> Result<Vec<Vec<usize>>, BixverseErrors> {
    if ef_search / no_neighbours <= 2 {
        println!(
            "[!WARNING!] Your 'ef_search' is set to {} for k {}. 'ef_search' should be 2 to 4x 'k'!",
            ef_search, no_neighbours
        )
    }

    let (res, index) = build_and_query_knn(
        no_neighbours,
        verbose,
        || {
            Ok(build_hnsw_index(
                mat,
                m,
                ef_const,
                dist_metric,
                seed,
                verbose,
            ))
        },
        |idx| query_hnsw_self(idx, no_neighbours + 1, ef_search, false, true),
        "HNSW",
    )?;

    if validate_index && verbose {
        let recall = index.validate_index(no_neighbours, seed, None)?;
        println!(
            "Recall of approximate nearest neighbours search in random subset: {:.2}",
            recall
        );
    }

    Ok(res)
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
/// * `dist_metric` - Distance metric to use. One of `"euclidean"` or
///   `"cosine"`.
/// * `no_neighbours` - Number of neighbours for the KNN graph.
/// * `n_trees` - Number of trees to use for the search.
/// * `search_budget` - Optional search budget per given query. If not provided,
///   it will use `k * n_trees * 20`.
/// * `seed` - Seed for the Annoy algorithm
/// * `validate_index` - Shall the index be validated with an exhaustive search.
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The k-nearest neighbours based on the Annoy algorithm. Function does not
/// return self.
#[allow(clippy::too_many_arguments)]
pub fn generate_knn_annoy(
    mat: MatRef<f32>,
    dist_metric: &str,
    no_neighbours: usize,
    n_trees: usize,
    search_budget: Option<usize>,
    seed: usize,
    validate_index: bool,
    verbose: bool,
) -> Result<Vec<Vec<usize>>, BixverseErrors> {
    let (res, index) = build_and_query_knn(
        no_neighbours,
        verbose,
        || build_annoy_index(mat, dist_metric, n_trees, seed),
        |idx| query_annoy_self(idx, no_neighbours + 1, search_budget, false, verbose),
        "Annoy",
    )?;

    if validate_index && verbose {
        let recall = index.validate_index(no_neighbours, seed, None)?;
        println!(
            "Recall of approximate nearest neighbours search in random subset: {:.2}",
            recall
        );
    }

    Ok(res)
}

/// Get the kNN graph based on IVF
///
/// This function generates the kNN graph based via an approximate nearest
/// neighbour search based on the IVF. The algorithm will use cluster the data
/// via k-means and probe n_probe clusters.
///
/// ### Params
///
/// * `mat` - Matrix in which rows represent the samples and columns the
///   respective embeddings for that sample
/// * `dist_metric` - The distance metric to use. One of `"euclidean"` or
///   `"cosine"`.
/// * `no_neighbours` - Number of neighbours for the KNN graph.
/// * `n_list` - Number of clusters/lists to generate. If None, will query
///   `sqrt(n)`.
/// * `n_probe` - Number of clusters/lists to query. If None, will query
///   `sqrt(n_list)`.
/// * `seed` - Seed for the IVF algorithm
/// * `validate_index` - Shall the index be validated with an exhaustive search.
/// * `verbose` - Controls verbosity of the algorithm
///
/// ### Returns
///
/// The k-nearest neighbours based on the IVF algorithm. Function does not
/// return self.
#[allow(clippy::too_many_arguments)]
pub fn generate_knn_ivf(
    mat: MatRef<f32>,
    dist_metric: &str,
    no_neighbours: usize,
    n_list: Option<usize>,
    n_probe: Option<usize>,
    seed: usize,
    validate_index: bool,
    verbose: bool,
) -> Result<Vec<Vec<usize>>, BixverseErrors> {
    let (res, index) = build_and_query_knn(
        no_neighbours,
        verbose,
        || build_ivf_index(mat, n_list, None, dist_metric, seed, verbose),
        |idx| query_ivf_self(idx, no_neighbours + 1, n_probe, false, verbose),
        "IVF",
    )?;

    if validate_index && verbose {
        let recall = index.validate_index(no_neighbours, seed, None)?;
        println!(
            "Recall of approximate nearest neighbours search in random subset: {:.2}",
            recall
        );
    }

    Ok(res)
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
/// * `extract_knn` - Hand back the descent graph directly rather than beam
///   searching it. Skips the query pass entirely, at the cost of some recall
///   and possibly short rows.
/// * `seed` - Seed for the NN Descent algorithm
/// * `validate_index` - Shall the index be validated with an exhaustive search.
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
    extract_knn: bool,
    seed: usize,
    validate_index: bool,
    verbose: bool,
) -> Result<Vec<Vec<usize>>, BixverseErrors> {
    if !extract_knn && ef_budget.is_none() && no_neighbours > 150 {
        println!(
            "[WARNING!] Your 'ef_budget' is set to auto ((k * 2).clamp(50, 200)) for k {}. 'ef_search' should be 2 to 4x 'k'",
            no_neighbours
        )
    }

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
        |idx| {
            if extract_knn {
                extract_nndescent_knn(idx, Some(no_neighbours + 1), true, false)
            } else {
                query_nndescent_self(idx, no_neighbours + 1, ef_budget, false, verbose)
            }
        },
        if extract_knn {
            "NNDescent (graph extraction)"
        } else {
            "NNDescent"
        },
    )?;

    if validate_index && verbose {
        let recall = index.validate_index(no_neighbours, seed, None)?;
        println!(
            "Recall of approximate nearest neighbours search in random subset: {:.2}",
            recall
        );
    }

    Ok(res)
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
) -> Result<Vec<Vec<usize>>, BixverseErrors> {
    let (res, _) = build_and_query_knn(
        no_neighbours,
        verbose,
        || Ok(build_exhaustive_index(mat, dist_metric)),
        |idx| query_exhaustive_self(idx, no_neighbours + 1, false, verbose),
        "exhaustive linear search",
    )?;

    Ok(res)
}

/// Get the kNN graph based on KmKnn
///
/// This function generates the kNN graph based on the k-means kNN (KmKnn)
/// algorithm. It provides the quality of an exhaustive search, but is much
/// faster.
///
/// ### Params
///
/// * `mat` - Matrix in which rows represent the samples and columns the
///   respective embeddings for that sample
/// * `dist_metric` - The distance metric to use. One of `"euclidean"` or
///   `"cosine"`.
/// * `no_neighbours` - Number of neighbours for the KNN graph.
/// * `n_list` - Number of clusters/lists to generate. If None, will query
///   `sqrt(n)`.
/// * `seed` - Seed for the NN Descent algorithm
/// * `validate_index` - Shall the index be validated with an exhaustive search.
/// * `verbose` - Controls verbosity of the algorithm
///
/// ### Returns
///
/// The k-nearest neighbours based on the NNDescent algorithm. Function does not
/// return self.
#[allow(clippy::too_many_arguments)]
pub fn generate_knn_kmknn(
    mat: MatRef<f32>,
    dist_metric: &str,
    no_neighbours: usize,
    n_list: Option<usize>,
    seed: usize,
    verbose: bool,
) -> Result<Vec<Vec<usize>>, BixverseErrors> {
    let (res, _) = build_and_query_knn(
        no_neighbours,
        verbose,
        || build_kmknn_index(mat, dist_metric, n_list, None, seed, verbose),
        |idx| query_kmknn_self(idx, no_neighbours + 1, false, verbose),
        "KmKnn",
    )?;

    Ok(res)
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
/// * `validate_index` - Shall the index be validated with an exhaustive search.
/// * `seed` - Seed for reproducibility
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// Tuple of `(indices of nearest neighbours, distances to these neighbours)`.
/// The distances are true distances: [`to_true_distances`] has already run.
pub fn generate_knn_with_dist(
    embd: MatRef<f32>,
    knn_params: &KnnParams,
    return_dist: bool,
    validate_index: bool,
    seed: usize,
    verbose: bool,
) -> ScKnnOptionVersion {
    fn remove_self(
        mut indices: Vec<Vec<usize>>,
        distances: Option<Vec<Vec<f32>>>,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<f32>>>) {
        // The extraction path can leave a row empty where the descent never
        // filled it, which the query paths never do, so this cannot assume a
        // self-edge is there to strip.
        for idx_vec in indices.iter_mut() {
            if !idx_vec.is_empty() {
                idx_vec.remove(0);
            }
        }
        let distances = distances.map(|mut dists| {
            for dist_vec in dists.iter_mut() {
                if !dist_vec.is_empty() {
                    dist_vec.remove(0);
                }
            }
            dists
        });
        (indices, distances)
    }

    fn timed<T>(name: &str, verbose: bool, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = f();
        if verbose {
            println!("{}: {:.2?}", name, start.elapsed());
        }
        result
    }

    let knn_method = parse_knn_method(&knn_params.knn_method).unwrap_or_default();
    warn_unsupported_extract(knn_params.extract_knn, knn_method);
    let k_plus_one = knn_params.k + 1;

    let (indices, distances) = match knn_method {
        KnnSearch::Annoy => {
            let index = timed("Generated Annoy index", verbose, || {
                build_annoy_index(embd, &knn_params.ann_dist, knn_params.n_tree, seed)
            })?;
            let (indices, distances) = timed("Queried Annoy index", verbose, || {
                query_annoy_self(
                    &index,
                    k_plus_one,
                    knn_params.search_budget,
                    return_dist,
                    verbose,
                )
            })?;
            if validate_index && verbose {
                let recall = index.validate_index(k_plus_one, seed, None)?;
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

            if knn_params.ef_search / k_plus_one <= 2 {
                println!(
                    "[WARNING!] Your 'ef_search' is set to {} for k {}. 'ef_search' should be 2 to 4x 'k'!",
                    knn_params.ef_search, k_plus_one
                )
            }

            let (indices, distances) = timed("Queried HNSW index", verbose, || {
                query_hnsw_self(
                    &index,
                    k_plus_one,
                    knn_params.ef_search,
                    return_dist,
                    verbose,
                )
            })?;
            if validate_index && verbose {
                let recall = index.validate_index(k_plus_one, seed, None)?;
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
            })?;

            if !knn_params.extract_knn && knn_params.ef_budget.is_none() && k_plus_one > 150 {
                println!(
                    "[WARNING!] Your 'ef_budget' is set to auto ((k * 2).clamp(50, 200)) for k {}. 'ef_search' should be 2 to 4x 'k'",
                    k_plus_one
                )
            }

            let (indices, distances) = if knn_params.extract_knn {
                timed("Extracted NNDescent graph", verbose, || {
                    extract_nndescent_knn(&index, Some(k_plus_one), true, true)
                })?
            } else {
                timed("Queried NNDescent index", verbose, || {
                    query_nndescent_self(&index, k_plus_one, knn_params.ef_budget, true, verbose)
                })?
            };
            if validate_index && verbose {
                let recall = index.validate_index(k_plus_one, seed, None)?;
                println!(
                    "Recall of approximate nearest neighbours search in random subset: {:.2}",
                    recall
                );
            }
            (indices, distances)
        }
        KnnSearch::Exhaustive => {
            let index = timed("Generated Exhaustive index", verbose, || {
                build_exhaustive_index(embd, &knn_params.ann_dist)
            });
            timed("Queried Exhaustive index", verbose, || {
                query_exhaustive_self(&index, k_plus_one, true, verbose)
            })?
        }
        KnnSearch::KmKnn => {
            let index = timed("Generated KmKnn index", verbose, || {
                build_kmknn_index(
                    embd,
                    &knn_params.ann_dist,
                    knn_params.n_list,
                    None,
                    seed,
                    verbose,
                )
            })?;
            timed("Queried KmKnn index", verbose, || {
                query_kmknn_self(&index, k_plus_one, true, verbose)
            })?
        }
        KnnSearch::Ivf => {
            let index = timed("Generated IVF index", verbose, || {
                build_ivf_index(
                    embd,
                    knn_params.n_list,
                    None,
                    &knn_params.ann_dist,
                    seed,
                    verbose,
                )
            })?;
            let (indices, distances) = timed("Queried IVF index", verbose, || {
                query_ivf_self(&index, k_plus_one, knn_params.n_probe, true, verbose)
            })?;
            if validate_index && verbose {
                let recall = index.validate_index(k_plus_one, seed, None)?;
                println!(
                    "Recall of approximate nearest neighbours search in random subset: {:.2}",
                    recall
                );
            }
            (indices, distances)
        }
    };

    let (indices, mut distances) = remove_self(indices, distances);
    if let Some(dists) = distances.as_mut() {
        to_true_distances(dists, &knn_params.ann_dist);
    }

    Ok((indices, distances))
}

///////////
// Other //
///////////

/// Compare kNN graphs
///
/// ### Params
///
/// * `a` - The first kNN graph in form samples x neighbour indices
/// * `b` - The second kNN graph in form samples x neighbour indices
///
/// ### Returns
///
/// Number of intersecting neighbours across the two.
pub fn compare_knn_graphs(a: MatRef<i32>, b: MatRef<i32>) -> Vec<i32> {
    assert_eq!(a.nrows(), b.nrows());

    (0..a.nrows())
        .into_par_iter()
        .map(|row| {
            let set: FxHashSet<i32> = (0..a.ncols()).map(|j| a[(row, j)]).collect();
            (0..b.ncols())
                .filter(|&j| set.contains(&b[(row, j)]))
                .count() as i32
        })
        .collect()
}

///////////////////////
// Neighbour kernels //
///////////////////////

/// Gaussian-kernel weights on the neighbour distances.
///
/// `w_ij = exp(-d_ij^2 / sigma_i^2)`, row-normalised to sum to one, with the
/// kernel width `sigma_i` set to node `i`'s `ceil(k / neighborhood_factor)`-th
/// smallest neighbour distance. A zero width, which happens when that many
/// neighbours sit exactly on top of the node, is replaced by one, and a row
/// whose weights all underflow is left alone rather than divided by zero.
///
/// ### Params
///
/// * `distances` - True neighbour distances per node, ascending, self excluded
/// * `neighborhood_factor` - Divisor picking which neighbour sets the width
///
/// ### Returns
///
/// One weight per neighbour, in the same layout as `distances`.
///
/// ### References
///
/// DeTomaso and Yosef, Cell Systems, 2021 (`hotspot/knn.py::compute_weights`)
pub fn knn_distance_weights(distances: &[Vec<f32>], neighborhood_factor: f32) -> Vec<Vec<f32>> {
    distances
        .par_iter()
        .map(|row| {
            if row.is_empty() {
                return Vec::new();
            }

            let radius =
                ((row.len() as f32 / neighborhood_factor).ceil() as usize).clamp(1, row.len());

            // everything below works in `d^2`, so the width is squared once here
            let sigma_sq = row[radius - 1] * row[radius - 1];
            let sigma_sq = if sigma_sq == 0.0 { 1.0 } else { sigma_sq };

            let mut weights: Vec<f32> = row.iter().map(|&d| (-(d * d) / sigma_sq).exp()).collect();

            let total: f32 = weights.iter().sum();
            if total != 0.0 {
                for w in weights.iter_mut() {
                    *w /= total;
                }
            }

            weights
        })
        .collect()
}

/// Turn the distances an `ann-search-rs` query hands back into true distances.
///
/// The crate folds `"euclidean"` and `"l2"` onto [`Dist::SquaredEuclidean`] and
/// never takes the square root, so every path in this crate that receives
/// neighbour distances applies it once, here. Past this boundary a distance is
/// a distance, and no kernel needs to ask which form it is holding. `"cosine"`
/// and `"manhattan"` already come back unsquared, so this is a no-op for them.
///
/// ### Params
///
/// * `distances` - Neighbour distances per node, rewritten in place
/// * `ann_dist` - Metric name, as carried in `KnnParams::ann_dist`
pub fn to_true_distances(distances: &mut [Vec<f32>], ann_dist: &str) {
    if !matches!(parse_ann_dist(ann_dist), Some(Dist::SquaredEuclidean)) {
        return;
    }

    distances
        .par_iter_mut()
        .for_each(|row| row.iter_mut().for_each(|d| *d = d.max(0.0).sqrt()));
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Values taken from `hotspot/knn.py::compute_weights` run on the same
    /// input: `radius_ii = ceil(6 / 3) = 2`, so the kernel width is the second
    /// neighbour distance. Rows 0 and 2 are the same distances scaled by two,
    /// and the kernel is scale-invariant, so they have to come out identical.
    #[test]
    fn test_knn_weights_match_the_reference_kernel() {
        let distances = vec![
            vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            vec![0.0; 6],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ];

        let expected = [
            0.612_048_74,
            0.289_111_36,
            0.082_831_79,
            0.014_394_007,
            0.001_517_117_2,
            0.000_096_986_055,
        ];

        let weights = knn_distance_weights(&distances, 3.0);

        for row in [0, 2] {
            for (got, want) in weights[row].iter().zip(expected.iter()) {
                assert_relative_eq!(got, want, epsilon = 1e-6);
            }
        }

        // a zero kernel width falls back to one, leaving every weight equal
        for &w in &weights[1] {
            assert_relative_eq!(w, 1.0 / 6.0, epsilon = 1e-6);
        }

        // every row sums to one
        for row in &weights {
            assert_relative_eq!(row.iter().sum::<f32>(), 1.0, epsilon = 1e-6);
        }
    }

    /// The kernel width tracks `neighborhood_factor`, so a wider neighbourhood
    /// has to spread the weight further out.
    #[test]
    fn test_neighborhood_factor_widens_the_kernel() {
        let distances = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];

        let tight = knn_distance_weights(&distances, 6.0);
        let wide = knn_distance_weights(&distances, 1.0);

        // ceil(6/6) = 1 -> sigma = 1.0, ceil(6/1) = 6 -> sigma = 6.0
        assert!(tight[0][0] > wide[0][0]);
        assert!(tight[0][5] < wide[0][5]);
    }

    /// Deterministic clustered data, `n` points in `dim` dimensions spread over
    /// `n_clusters` well-separated blobs.
    fn clustered_data(n: usize, dim: usize, n_clusters: usize) -> faer::Mat<f32> {
        let hash = |a: usize, b: usize| -> f32 {
            let mut h = (a as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
            h ^= h >> 29;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 32;
            ((h >> 40) as f32) / 16_777_216.0 - 0.5
        };

        faer::Mat::<f32>::from_fn(n, dim, |i, j| {
            let cluster = i % n_clusters;
            hash(cluster + 1000, j) * 20.0 + hash(i, j) * 2.0
        })
    }

    /// `extract_knn` reshapes the graph the descent already built rather than
    /// beam searching it, so it trades a little recall for skipping the query
    /// pass. It still has to agree with the beam search on most neighbours,
    /// never return self, and never hand back more than `k`.
    #[test]
    fn test_nndescent_extract_agrees_with_the_query_path() {
        let (n, k) = (800usize, 10usize);
        let data = clustered_data(n, 16, 8);

        let base = KnnParams {
            knn_method: "nndescent".to_string(),
            ann_dist: "euclidean".to_string(),
            k,
            ..Default::default()
        };
        let extract = KnnParams {
            extract_knn: true,
            ..base.clone()
        };

        let (queried, _) = generate_knn_with_dist(data.as_ref(), &base, false, false, 42, false)
            .expect("the beam search path must succeed");
        let (extracted, dists) =
            generate_knn_with_dist(data.as_ref(), &extract, true, false, 42, false)
                .expect("the extraction path must succeed");

        assert_eq!(extracted.len(), n);
        let dists = dists.expect("distances were requested");

        let mut overlap = 0usize;
        for (i, (got, want)) in extracted.iter().zip(queried.iter()).enumerate() {
            assert!(got.len() <= k, "row {} returned {} > {}", i, got.len(), k);
            assert!(!got.contains(&i), "row {} contains itself", i);
            assert_eq!(got.len(), dists[i].len(), "row {} shape mismatch", i);

            let truth: FxHashSet<usize> = want.iter().copied().collect();
            overlap += got.iter().filter(|x| truth.contains(x)).count();
        }

        let recall = overlap as f32 / (n * k) as f32;
        assert!(
            recall > 0.9,
            "extraction recall was {:.4} against the beam search",
            recall
        );
    }

    /// `"euclidean"` and `"l2"` come back squared from `ann-search-rs` and get
    /// rooted here; every other metric is already a distance and must be left
    /// alone. A name that does not parse is left alone too.
    #[test]
    fn test_to_true_distances_only_roots_the_squared_metric() {
        let squared = || vec![vec![4.0f32, 9.0, 16.0], vec![0.0, 25.0]];

        for metric in ["euclidean", "l2", "EUCLIDEAN"] {
            let mut d = squared();
            to_true_distances(&mut d, metric);
            assert_eq!(d, vec![vec![2.0, 3.0, 4.0], vec![0.0, 5.0]], "{}", metric);
        }

        for metric in ["cosine", "manhattan", "not a metric"] {
            let mut d = squared();
            to_true_distances(&mut d, metric);
            assert_eq!(d, squared(), "{}", metric);
        }
    }
}
