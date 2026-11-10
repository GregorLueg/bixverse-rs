//! Contains the single cell-related kNN functions. Wrappers to generate
//! the kNN graphs (with and without distances).

use ann_search_rs::prelude::{KNN_GRAPH_EMPTY, KnnGraph, NNDescentKnnParams};
use ann_search_rs::utils::KnnValidation;
use ann_search_rs::utils::dist::{Dist, SimdDistance, parse_ann_dist};
use ann_search_rs::*;
use faer::{Mat, MatRef, RowRef};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Deserializer, de::Error as _};
use std::time::Instant;

use crate::prelude::*;
use bixverse_core::core::math::sparse::coo_to_csr;

///////////
// Types //
///////////

/// Single cell Knn results with optional distances
pub type ScKnnOptionVersion = Result<(Vec<Vec<usize>>, Option<Vec<Vec<f32>>>), ScErrors>;

/// Single cell Knn results with distances
pub type ScKnnResults<E = ScErrors> = Result<(Vec<Vec<usize>>, Vec<Vec<f32>>), E>;

/// A kNN graph as `(indices, distances)`, one row per cell, self excluded
pub type KnnRows = (Vec<Vec<usize>>, Vec<Vec<f32>>);

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
    /// NNDescent. A self-kNN is the descent graph itself, built without an
    /// index and handed back without a search, short rows topped up exactly
    /// (see [`top_up_short_rows`]). A cross-query builds the NNDescent index
    /// and beam searches it.
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
/// The matching [`KnnSearch`], or `None` if the string is unrecognised.
/// `"cagra"` is GPU / MLX only and returns `None` here.
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

/// Deserialise a kNN method name, checked through [parse_knn_method]
///
/// ### Params
///
/// * `d` - The deserialiser, holding a string
///
/// ### Returns
///
/// The name as given, or an error naming the unrecognised value.
pub fn de_knn_method<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let s = String::deserialize(d)?;
    match parse_knn_method(&s) {
        Some(_) => Ok(s),
        None => Err(D::Error::custom(format!(
            "knn_method must be one of 'kmknn', 'annoy', 'hnsw', 'nndescent', \
             'exhaustive' or 'ivf' ('cagra' is GPU / MLX only), got '{s}'"
        ))),
    }
}

/// Deserialise a distance metric name, checked through `parse_ann_dist`
///
/// ### Params
///
/// * `d` - The deserialiser, holding a string
///
/// ### Returns
///
/// The name as given, or an error naming the unrecognised value.
pub fn de_ann_dist<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let s = String::deserialize(d)?;
    match parse_ann_dist(&s) {
        Some(_) => Ok(s),
        None => Err(D::Error::custom(format!(
            "distance must be one of 'euclidean', 'cosine' or 'manhattan', got '{s}'"
        ))),
    }
}

////////////
// Params //
////////////

/// KnnParams
///
/// Contains the parameters for the kNN searches used in the single cell parts
/// of this crate
///
/// Deserialises from the kNN keys of the R `params_knn_defaults()` list
/// (`n_trees` for [`KnnParams::n_tree`]); missing keys take the defaults.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct KnnParams {
    ///  Which of the kNN methods to use. One of `"annoy"`, `"hnsw"`, `"ivf"`,
    /// `"kmknn"`, `"exhaustive"` or `"nndescent"` are supported for now.
    #[serde(deserialize_with = "de_knn_method")]
    pub knn_method: String,
    /// Distance metric to use. One of `"euclidean"` or `"cosine"`.
    #[serde(deserialize_with = "de_ann_dist")]
    pub ann_dist: String,
    /// Number of neighbours to return
    pub k: usize,
    /// Annoy: Number of trees to build
    #[serde(rename = "n_trees")]
    pub n_tree: usize,
    /// Annoy: optional search budget. If not provided, will default to k * 20
    /// per tree.
    pub search_budget: Option<usize>,
    /// NNDescent: diversification probability after generation of the graph.
    pub diversify_prob: f32,
    /// NNDescent: convergence criterium. If less than these
    /// percentage of neighbours have been udpated, the algorithm counts as
    /// converged.
    pub delta: f32,
    /// NNDescent: optional beam search budget for querying.
    pub ef_budget: Option<usize>,

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

/// Reshape a flat [`KnnGraph`] into one row per sample.
///
/// Empty slots ([`KNN_GRAPH_EMPTY`]) are dropped, so a row the descent never
/// filled comes back shorter than `k`.
///
/// ### Params
///
/// * `graph` - The flat graph from one of the graph-only builders
///
/// ### Returns
///
/// Tuple of `(neighbour indices, distances if the graph kept them)`.
pub fn knn_graph_to_rows(graph: KnnGraph<f32>) -> (Vec<Vec<usize>>, Option<Vec<Vec<f32>>>) {
    let k = graph.k.max(1);
    let indices: Vec<Vec<usize>> = graph
        .indices
        .par_chunks(k)
        .map(|row| {
            row.iter()
                .take_while(|&&id| id != KNN_GRAPH_EMPTY)
                .map(|&id| id as usize)
                .collect()
        })
        .collect();
    let distances = graph.distances.map(|dists| {
        dists
            .par_chunks(k)
            .zip(indices.par_iter())
            .map(|(row, ids)| row[..ids.len()].to_vec())
            .collect()
    });

    (indices, distances)
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
) -> Result<Vec<Vec<usize>>, ScErrors> {
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
            build_hnsw_index(
                mat,
                m,
                ef_const,
                dist_metric,
                seed,
                verbosity_from_flag(verbose),
            )
        },
        |idx| {
            query_hnsw_self(
                idx,
                no_neighbours + 1,
                ef_search,
                false,
                verbosity_from_flag(verbose),
            )
        },
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
) -> Result<Vec<Vec<usize>>, ScErrors> {
    let (res, index) = build_and_query_knn(
        no_neighbours,
        verbose,
        || {
            build_annoy_index(
                mat,
                dist_metric,
                n_trees,
                seed,
                verbosity_from_flag(verbose),
            )
        },
        |idx| {
            query_annoy_self(
                idx,
                no_neighbours + 1,
                search_budget,
                false,
                verbosity_from_flag(verbose),
            )
        },
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
) -> Result<Vec<Vec<usize>>, ScErrors> {
    let (res, index) = build_and_query_knn(
        no_neighbours,
        verbose,
        || {
            build_ivf_index(
                mat,
                n_list,
                None,
                dist_metric,
                seed,
                verbosity_from_flag(verbose),
            )
        },
        |idx| {
            query_ivf_self(
                idx,
                no_neighbours + 1,
                n_probe,
                false,
                verbosity_from_flag(verbose),
            )
        },
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
    seed: usize,
    validate_index: bool,
    verbose: bool,
) -> Result<Vec<Vec<usize>>, ScErrors> {
    if ef_budget.is_none() && no_neighbours > 150 {
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
                verbosity_from_flag(verbose),
            )
        },
        |idx| {
            query_nndescent_self(
                idx,
                no_neighbours + 1,
                ef_budget,
                false,
                verbosity_from_flag(verbose),
            )
        },
        "NNDescent",
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
) -> Result<Vec<Vec<usize>>, ScErrors> {
    let (res, _) = build_and_query_knn(
        no_neighbours,
        verbose,
        || {
            Ok(build_exhaustive_index(
                mat,
                dist_metric,
                verbosity_from_flag(verbose),
            ))
        },
        |idx| query_exhaustive_self(idx, no_neighbours + 1, false, verbosity_from_flag(verbose)),
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
) -> Result<Vec<Vec<usize>>, ScErrors> {
    let (res, _) = build_and_query_knn(
        no_neighbours,
        verbose,
        || {
            build_kmknn_index(
                mat,
                dist_metric,
                n_list,
                None,
                seed,
                verbosity_from_flag(verbose),
            )
        },
        |idx| query_kmknn_self(idx, no_neighbours + 1, false, verbosity_from_flag(verbose)),
        "KmKnn",
    )?;

    Ok(res)
}

/// Get the kNN graph straight from NN-Descent
///
/// Matrix in, graph out: no index is kept and no search runs, the graph the
/// descent converges to is handed back as is. Seeding and descent match
/// [`generate_knn_nndescent`], minus the beam search over the result.
///
/// ### Params
///
/// * `mat` - Matrix in which rows represent the samples and columns the
///   respective embeddings for that sample
/// * `dist_metric` - The distance metric to use. One of `"euclidean"` or
///   `"cosine"`.
/// * `no_neighbours` - Number of neighbours for the KNN graph.
/// * `delta` - Early stop criterium for the algorithm.
/// * `seed` - Seed for the NN Descent algorithm
/// * `verbose` - Controls verbosity of the algorithm
///
/// ### Returns
///
/// The k-nearest neighbours, self excluded. Rows the descent left short are
/// topped up with [`top_up_short_rows`], so every row holds `no_neighbours`
/// (fewer only when there are not that many other cells).
pub fn generate_knn_nndescent_graph(
    mat: MatRef<f32>,
    dist_metric: &str,
    no_neighbours: usize,
    delta: f32,
    seed: usize,
    verbose: bool,
) -> Result<Vec<Vec<usize>>, ScErrors> {
    let params =
        NNDescentKnnParams::new(no_neighbours, None, None, None, delta as f64, false, false);
    let graph = build_knn_graph_nndescent(
        mat,
        dist_metric,
        Some(params),
        seed,
        verbosity_from_flag(verbose),
    )?;

    let mut indices = knn_graph_to_rows(graph).0;
    top_up_short_rows(mat, &mut indices, None, no_neighbours, dist_metric, verbose)?;

    Ok(indices)
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
    // By index, not position: with an exact duplicate at distance 0 the
    // duplicate can sort ahead of self. Rows that miss self lose the furthest.
    fn remove_self(
        mut indices: Vec<Vec<usize>>,
        mut distances: Option<Vec<Vec<f32>>>,
        k: usize,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<f32>>>) {
        for (i, idx_vec) in indices.iter_mut().enumerate() {
            let pos = idx_vec.iter().position(|&j| j == i);
            if let Some(p) = pos {
                idx_vec.remove(p);
            }
            idx_vec.truncate(k);
            if let Some(dists) = distances.as_mut() {
                if let Some(p) = pos {
                    dists[i].remove(p);
                }
                dists[i].truncate(k);
            }
        }
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
    let verbosity = verbosity_from_flag(verbose);
    let k_plus_one = knn_params.k + 1;

    let (indices, distances) = match knn_method {
        KnnSearch::Annoy => {
            let index = timed("Generated Annoy index", verbose, || {
                build_annoy_index(
                    embd,
                    &knn_params.ann_dist,
                    knn_params.n_tree,
                    seed,
                    verbosity,
                )
            })?;
            let (indices, distances) = timed("Queried Annoy index", verbose, || {
                query_annoy_self(
                    &index,
                    k_plus_one,
                    knn_params.search_budget,
                    return_dist,
                    verbosity,
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
                    verbosity,
                )
            })?;

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
                    verbosity,
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
            // The self-edge is requested so these rows go through the same
            // `remove_self` as every query path.
            let params = NNDescentKnnParams::new(
                knn_params.k,
                None,
                None,
                None,
                knn_params.delta as f64,
                true,
                true,
            );
            let graph = timed("Generated NNDescent graph", verbose, || {
                build_knn_graph_nndescent(embd, &knn_params.ann_dist, Some(params), seed, verbosity)
            })?;
            knn_graph_to_rows(graph)
        }
        KnnSearch::Exhaustive => {
            let index = timed("Generated Exhaustive index", verbose, || {
                build_exhaustive_index(embd, &knn_params.ann_dist, verbosity)
            });
            timed("Queried Exhaustive index", verbose, || {
                query_exhaustive_self(&index, k_plus_one, true, verbosity)
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
                    verbosity,
                )
            })?;
            timed("Queried KmKnn index", verbose, || {
                query_kmknn_self(&index, k_plus_one, true, verbosity)
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
                    verbosity,
                )
            })?;
            let (indices, distances) = timed("Queried IVF index", verbose, || {
                query_ivf_self(&index, k_plus_one, knn_params.n_probe, true, verbosity)
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

    let (mut indices, mut distances) = remove_self(indices, distances, knn_params.k);
    if knn_method == KnnSearch::NNDescent {
        top_up_short_rows(
            embd,
            &mut indices,
            distances.as_deref_mut(),
            knn_params.k,
            &knn_params.ann_dist,
            verbose,
        )?;
    }
    if let Some(dists) = distances.as_mut() {
        to_true_distances(dists, &knn_params.ann_dist);
    }

    Ok((indices, distances))
}

/// A supplied kNN graph, or one searched on the embedding
///
/// What the HotSpot, VISION and MELD entry points do with an optional cached
/// graph: take it as is, else search `embd` without validation.
///
/// ### Params
///
/// * `knn` - Optional `(indices, distances)` without self. The caller vouches
///   that the rows line up with `embd`.
/// * `embd` - Cells x features embedding to search when `knn` is `None`
/// * `knn_params` - The search parameters
/// * `seed` - Seed for the index build
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(indices, distances)`, self excluded, true distances.
pub fn knn_or_generate(
    knn: Option<KnnRows>,
    embd: MatRef<f32>,
    knn_params: &KnnParams,
    seed: usize,
    verbose: bool,
) -> ScKnnResults {
    match knn {
        Some(knn) => {
            if verbose {
                println!("Using provided kNN graph.");
            }
            Ok(knn)
        }
        None => {
            if verbose {
                println!("Generating a kNN graph from scratch.");
            }
            let (indices, dist) =
                generate_knn_with_dist(embd, knn_params, true, false, seed, verbose)?;
            Ok((indices, dist.expect("return_dist is set")))
        }
    }
}

/// Fill the rows a graph-only NN-descent build left short of `k`
///
/// The descent can leave a row short where it never filled it. Only those
/// rows are re-queried, exactly, against the whole embedding, so the cost
/// scales with the number of short rows rather than with `n`. A filled row is
/// replaced wholesale by its exact neighbours.
///
/// ### Params
///
/// * `embd` - The embedding the graph was built on, cells x features
/// * `indices` - Neighbour rows, self excluded, replaced in place where short
/// * `distances` - Optional distances alongside, in the index metric (squared
///   for Euclidean, so before [`to_true_distances`])
/// * `k` - Target row length
/// * `ann_dist` - Distance metric name, as handed to the graph build
/// * `verbose` - Print how many rows were filled
///
/// ### Returns
///
/// The number of rows that were filled.
pub fn top_up_short_rows(
    embd: MatRef<f32>,
    indices: &mut [Vec<usize>],
    mut distances: Option<&mut [Vec<f32>]>,
    k: usize,
    ann_dist: &str,
    verbose: bool,
) -> Result<usize, ScErrors> {
    let short: Vec<usize> = indices
        .iter()
        .enumerate()
        .filter(|(_, row)| row.len() < k)
        .map(|(i, _)| i)
        .collect();
    if short.is_empty() {
        return Ok(0);
    }

    let quiet = verbosity_from_flag(false);
    let query = Mat::<f32>::from_fn(short.len(), embd.ncols(), |r, c| embd[(short[r], c)]);
    let index = build_exhaustive_index(embd, ann_dist, quiet);
    let (rows, dists) = query_exhaustive_index(query.as_ref(), &index, k + 1, true, quiet)?;
    let dists = dists.expect("return_dist is set");

    for ((&i, row), dist) in short.iter().zip(rows).zip(dists) {
        let (row, dist): (Vec<usize>, Vec<f32>) = row
            .into_iter()
            .zip(dist)
            .filter(|&(j, _)| j != i)
            .take(k)
            .unzip();
        indices[i] = row;
        if let Some(d) = distances.as_mut() {
            d[i] = dist;
        }
    }

    if verbose {
        println!(
            "Filled {} kNN rows the NN-descent graph left short of k = {} with an exact search.",
            short.len(),
            k
        );
    }

    Ok(short.len())
}

/// Pad the rows a search left short of `k` neighbours
///
/// A short row repeats its last neighbour (and distance); an empty row
/// points at itself at distance 0. The searches here only leave rows short
/// when there are fewer than `k` other cells.
///
/// ### Params
///
/// * `indices` - Neighbour indices per node, padded in place
/// * `distances` - Optional distances per node, padded alongside
/// * `k` - Target row length
///
/// ### Returns
///
/// The number of rows that were padded.
pub fn pad_knn_rows(
    indices: &mut [Vec<usize>],
    mut distances: Option<&mut [Vec<f32>]>,
    k: usize,
) -> usize {
    let mut padded = 0;
    for (i, idx_row) in indices.iter_mut().enumerate() {
        if idx_row.len() >= k {
            continue;
        }
        padded += 1;
        let (fill_idx, fill_dist) = match (idx_row.last(), distances.as_ref()) {
            (Some(&last), Some(dists)) => (last, dists[i].last().copied().unwrap_or(0.0)),
            (Some(&last), None) => (last, 0.0),
            (None, _) => (i, 0.0),
        };
        idx_row.resize(k, fill_idx);
        if let Some(dists) = distances.as_mut() {
            dists[i].resize(k, fill_dist);
        }
    }
    padded
}

/// Recall and distance ratio of one kNN against a reference kNN
#[derive(Clone, Debug)]
pub struct KnnMetrics {
    /// Neighbours shared with the reference, per node
    pub all_matches: Vec<usize>,
    /// Summed query distance over summed reference distance, per node. Nodes
    /// whose reference distances sum to (almost) zero are skipped, so this
    /// can be shorter than `all_matches`.
    pub all_ratios: Vec<f64>,
    /// Mean of `matches / k` over all nodes
    pub final_recall: f64,
    /// Mean of `all_ratios`
    pub final_ratio: f64,
}

/// Reference distance sums at or below this are skipped in the ratio.
const KNN_RATIO_MIN_SUM: f64 = 1e-12;

/// Compare a kNN against a reference kNN
///
/// The per-node counts run in parallel; the means are summed in node order,
/// so the result does not depend on the thread count.
///
/// ### Params
///
/// * `idx_ref` - Reference neighbour indices, row-major, `k` per node
/// * `dist_ref` - Reference distances, same layout
/// * `idx_query` - Query neighbour indices, same layout
/// * `dist_query` - Query distances, same layout
/// * `k` - Neighbours per node
///
/// ### Returns
///
/// The [`KnnMetrics`], or an error if the layouts do not match.
pub fn compare_knn(
    idx_ref: &[u32],
    dist_ref: &[f32],
    idx_query: &[u32],
    dist_query: &[f32],
    k: usize,
) -> Result<KnnMetrics, ScErrors> {
    if k == 0
        || !idx_ref.len().is_multiple_of(k)
        || idx_ref.len() != idx_query.len()
        || dist_ref.len() != idx_ref.len()
        || dist_query.len() != idx_ref.len()
    {
        return Err(CoreErrors::InvalidArgument(format!(
            "both kNNs need the same number of nodes and k = {k} neighbours each, with \
             one distance per neighbour"
        ))
        .into());
    }

    let all_matches: Vec<usize> = idx_ref
        .par_chunks(k)
        .zip(idx_query.par_chunks(k))
        .map(|(a, b)| {
            let set: FxHashSet<u32> = a.iter().copied().collect();
            let query: FxHashSet<u32> = b.iter().copied().collect();
            query.intersection(&set).count()
        })
        .collect();
    let final_recall = all_matches
        .iter()
        .map(|&m| m as f64 / k as f64)
        .sum::<f64>()
        / all_matches.len() as f64;

    let all_ratios: Vec<f64> = dist_ref
        .chunks(k)
        .zip(dist_query.chunks(k))
        .filter_map(|(a, b)| {
            let sum_a: f64 = a.iter().map(|&x| x as f64).sum();
            let sum_b: f64 = b.iter().map(|&x| x as f64).sum();
            (sum_a > KNN_RATIO_MIN_SUM).then(|| sum_b / sum_a)
        })
        .collect();
    let final_ratio = all_ratios.iter().sum::<f64>() / all_ratios.len() as f64;

    Ok(KnnMetrics {
        all_matches,
        all_ratios,
        final_recall,
        final_ratio,
    })
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

    /// `"nndescent"` hands back the graph the descent converged to rather than
    /// beam searching an index. It still has to agree with the exact search on
    /// most neighbours, never return self, and fill every row to `k`.
    #[test]
    fn test_nndescent_graph_agrees_with_exhaustive() {
        let (n, k) = (800usize, 10usize);
        let data = clustered_data(n, 16, 8);

        let graph = KnnParams {
            knn_method: "nndescent".to_string(),
            ann_dist: "euclidean".to_string(),
            k,
            ..Default::default()
        };
        let exact = KnnParams {
            knn_method: "exhaustive".to_string(),
            ..graph.clone()
        };

        let (truth, _) = generate_knn_with_dist(data.as_ref(), &exact, false, false, 42, false)
            .expect("the exhaustive path must succeed");
        let (got, dists) = generate_knn_with_dist(data.as_ref(), &graph, true, false, 42, false)
            .expect("the graph path must succeed");

        assert_eq!(got.len(), n);
        let dists = dists.expect("distances were requested");

        let mut overlap = 0usize;
        for (i, (row, want)) in got.iter().zip(truth.iter()).enumerate() {
            assert_eq!(row.len(), k, "row {} holds {} != {}", i, row.len(), k);
            assert!(!row.contains(&i), "row {} contains itself", i);
            assert_eq!(row.len(), dists[i].len(), "row {} shape mismatch", i);

            let want: FxHashSet<usize> = want.iter().copied().collect();
            overlap += row.iter().filter(|x| want.contains(x)).count();
        }

        let recall = overlap as f32 / (n * k) as f32;
        assert!(
            recall > 0.9,
            "graph recall was {:.4} against exhaustive",
            recall
        );
    }

    /// Short rows, empty ones included, come back as the exact neighbours with
    /// their distances; full rows are left untouched.
    #[test]
    fn test_top_up_short_rows_fills_exactly() {
        let (n, k) = (300usize, 6usize);
        let data = clustered_data(n, 8, 4);
        // raw metric distances, as the graph path hands them to the top-up
        let index = build_exhaustive_index(data.as_ref(), "euclidean", verbosity_from_flag(false));
        let (raw_idx, raw_dist) =
            query_exhaustive_self(&index, k + 1, true, verbosity_from_flag(false))
                .expect("exhaustive self-query must succeed");
        let raw_dist = raw_dist.expect("distances were requested");
        let truth: Vec<Vec<usize>> = raw_idx.iter().map(|r| r[1..].to_vec()).collect();
        let truth_dist: Vec<Vec<f32>> = raw_dist.iter().map(|r| r[1..].to_vec()).collect();

        let mut indices = truth.clone();
        let mut dists = truth_dist.clone();
        indices[3].truncate(2);
        dists[3].truncate(2);
        indices[7].clear();
        dists[7].clear();
        // a full row with a wrong neighbour must not be touched
        indices[9][0] = 0;

        let filled = top_up_short_rows(
            data.as_ref(),
            &mut indices,
            Some(&mut dists),
            k,
            "euclidean",
            false,
        )
        .expect("the top-up must succeed");

        assert_eq!(filled, 2);
        for i in [3, 7] {
            assert_eq!(indices[i], truth[i], "row {} indices", i);
            for (got, want) in dists[i].iter().zip(&truth_dist[i]) {
                assert_relative_eq!(got, want, epsilon = 1e-5);
            }
        }
        assert_eq!(indices[9][0], 0);
    }

    /// Every cell has an exact duplicate, so self and its twin tie at distance
    /// 0 and either can come first. Self has to go, the twin has to stay.
    #[test]
    fn test_remove_self_with_exact_duplicates() {
        let (n, k) = (400usize, 5usize);
        let base = clustered_data(n / 2, 8, 4);
        let data = faer::Mat::<f32>::from_fn(n, 8, |i, j| base[(i / 2, j)]);

        for method in ["exhaustive", "kmknn", "hnsw", "annoy", "ivf"] {
            let params = KnnParams {
                knn_method: method.to_string(),
                ann_dist: "euclidean".to_string(),
                k,
                ..Default::default()
            };
            let (idx, dists) =
                generate_knn_with_dist(data.as_ref(), &params, true, false, 42, false)
                    .expect("the search must succeed");
            let dists = dists.expect("distances were requested");

            for (i, row) in idx.iter().enumerate() {
                assert_eq!(row.len(), k, "{}: row {} length", method, i);
                assert_eq!(dists[i].len(), k, "{}: row {} dist length", method, i);
                assert!(!row.contains(&i), "{}: row {} contains itself", method, i);
                if method == "exhaustive" || method == "kmknn" {
                    assert!(
                        row.contains(&(i ^ 1)),
                        "{}: row {} lost its twin",
                        method,
                        i
                    );
                }
            }
        }
    }

    /// The indices-only graph path (no self-edge requested upstream) and the
    /// with-distances path (self-edge requested, then stripped) must return the
    /// same rows for the same seed.
    #[test]
    fn test_nndescent_graph_paths_agree() {
        let (n, k) = (500usize, 8usize);
        let data = clustered_data(n, 12, 5);
        let params = KnnParams {
            knn_method: "nndescent".to_string(),
            ann_dist: "euclidean".to_string(),
            k,
            ..Default::default()
        };

        let direct =
            generate_knn_nndescent_graph(data.as_ref(), "euclidean", k, params.delta, 7, false)
                .expect("the indices-only graph path must succeed");
        let (with_dist, _) = generate_knn_with_dist(data.as_ref(), &params, true, false, 7, false)
            .expect("the with-distances graph path must succeed");

        assert_eq!(direct, with_dist);
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
