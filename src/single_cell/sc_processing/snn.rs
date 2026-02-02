use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

///////////
// Enums //
///////////

/// SNN similarity method
#[derive(Clone, Copy, Default)]
pub enum SnnSimilarityMethod {
    /// This will calculate the Jaccard similarity as weight
    #[default]
    Intersection,
    /// This will calculate the Rank version as a weight
    Rank,
}

/// Helper function to get the type of sNN similarity
///
/// ### Params
///
/// * `s` - Type of SNN similarity to use
///
/// ### Returns
///
/// Option of the SnnSimilarityMethod
pub fn get_snn_similiarity_method(s: &str) -> Option<SnnSimilarityMethod> {
    match s.to_lowercase().as_str() {
        "jaccard" => Some(SnnSimilarityMethod::Intersection),
        "rank" => Some(SnnSimilarityMethod::Rank),
        _ => None,
    }
}

///////////////////
// sNN functions //
///////////////////

/// Generate an sNN graph based on the kNN graph (full)
///
/// This version will compare all cells against all cells and generate an edge
/// if any neighbours are shared. This yields way denser graphs and is the
/// approach taken in the `bluster` R package to generate the sNN.
///
/// ### Params
///
/// * `knn_graph` - K-nearest neighbours data as a flat vector in column-major.
/// * `no_neighbours` - Number of neighbours in the kNN graph
/// * `pruning` - Below which Jaccard similarity to prune the edge. In this case
///   the weight is set to `0`.
/// * `method` - Which similarity method to use
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// A tuple with `(<edges>, <weights>)`. The edges are stored in a way that the
/// the first edge points goes from the first element to the second, the second
/// edge from the third to the fourth, etc.
pub fn generate_snn_full(
    flat_knn: &[usize],
    k: usize,
    n_samples: usize,
    pruning: f32,
    method: SnnSimilarityMethod,
    verbose: bool,
) -> (Vec<usize>, Vec<f32>) {
    let mut reverse_mappings: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_samples];

    let start_time = Instant::now();

    for i in 0..n_samples {
        reverse_mappings[i].push((i, 0));

        for neighbor_idx in 0..k {
            let neighbor = flat_knn[neighbor_idx * n_samples + i];
            reverse_mappings[neighbor].push((i, neighbor_idx + 1));
        }
    }

    let results: Vec<(usize, usize, f32)> = (0..n_samples)
        .into_par_iter()
        .flat_map(|j| {
            let mut scores = vec![0.0f32; n_samples];
            let mut added = Vec::new();

            for i in 0..=k {
                let cur_neighbor = if i == 0 {
                    j
                } else {
                    flat_knn[(i - 1) * n_samples + j]
                };

                for &(othernode, other_rank) in &reverse_mappings[cur_neighbor] {
                    if othernode < j {
                        match method {
                            SnnSimilarityMethod::Rank => {
                                let combined_rank = (i + other_rank) as f32;
                                if scores[othernode] == 0.0 {
                                    scores[othernode] = combined_rank;
                                    added.push(othernode);
                                } else if combined_rank < scores[othernode] {
                                    scores[othernode] = combined_rank;
                                }
                            }
                            SnnSimilarityMethod::Intersection => {
                                if scores[othernode] == 0.0 {
                                    added.push(othernode);
                                }
                                scores[othernode] += 1.0;
                            }
                        }
                    }
                }
            }

            added
                .into_iter()
                .filter_map(|othernode| {
                    let weight = match method {
                        SnnSimilarityMethod::Rank => {
                            let preliminary = k as f32 - scores[othernode] / 2.0;
                            let raw_weight = preliminary.max(1e-6);
                            raw_weight / k as f32
                        }
                        SnnSimilarityMethod::Intersection => {
                            scores[othernode] / (2.0 * (k as f32 + 1.0) - scores[othernode])
                        }
                    };

                    if weight >= pruning {
                        Some((j, othernode, weight))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut edges = Vec::with_capacity(results.len() * 2);
    let mut weights = Vec::with_capacity(results.len());

    for (i, j, weight) in results {
        edges.push(i);
        edges.push(j);
        weights.push(weight);
    }

    let end_snn = start_time.elapsed();

    if verbose {
        println!("Transformed kNN into a full sNN graph: {:.2?}", end_snn);
    }

    (edges, weights)
}

/// Generate an sNN graph based on the kNN graph (limited)
///
/// This version will only compare cells to the neighbouring cells and
/// deduplicate edges in taking the maximum weight between two given cells.
///
/// ### Params
///
/// * `knn_graph` - K-nearest neighbours data as a flat vector in column-major.
/// * `no_neighbours` - Number of neighbours in the kNN graph
/// * `pruning` - Below which Jaccard similarity to prune the edge. In this case
///   the weight is set to `0`.
/// * `method` - Which similarity method to use.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// A tuple with `(<edges>, <weights>)`. The edges are stored in a way that the
/// the first edge points goes from the first element to the second, the second
/// edge from the third to the fourth, etc.
pub fn generate_snn_limited(
    flat_knn: &[usize],
    k: usize,
    n_samples: usize,
    pruning: f32,
    method: SnnSimilarityMethod,
    verbose: bool,
) -> (Vec<usize>, Vec<f32>) {
    // We need to use a hashmap to store unique edges (smaller_idx, larger_idx)
    // -> weight
    let start_time = Instant::now();

    let edge_map: FxHashMap<(usize, usize), f32> = (0..n_samples)
        .into_par_iter()
        .flat_map(|i| {
            let mut edges = Vec::new();

            // only consider edges to this cell's k nearest neighbors
            for neighbor_idx in 0..k {
                let j = flat_knn[neighbor_idx * n_samples + i];

                // Calculate sNN similarity between cell i and its neighbor j
                let weight = match method {
                    SnnSimilarityMethod::Intersection => {
                        // Get neighbors of both cells
                        let neighbors_i: FxHashSet<usize> = (0..k)
                            .map(|idx| flat_knn[idx * n_samples + i])
                            .chain(std::iter::once(i)) // include self
                            .collect();

                        let neighbors_j: FxHashSet<usize> = (0..k)
                            .map(|idx| flat_knn[idx * n_samples + j])
                            .chain(std::iter::once(j)) // include self
                            .collect();

                        let intersection_count =
                            neighbors_i.intersection(&neighbors_j).count() as f32;
                        intersection_count / (2.0 * (k as f32 + 1.0) - intersection_count)
                        // Jaccard
                    }
                    SnnSimilarityMethod::Rank => {
                        // build ranks i
                        let mut ranks_i = FxHashMap::default();
                        ranks_i.insert(i, 0); // self at rank 0
                        for (rank, neighbor) in
                            (0..k).map(|idx| flat_knn[idx * n_samples + i]).enumerate()
                        {
                            ranks_i.insert(neighbor, rank + 1);
                        }

                        // build ranks j
                        let mut ranks_j = FxHashMap::default();
                        ranks_j.insert(j, 0); // self at rank 0
                        for (rank, neighbor) in
                            (0..k).map(|idx| flat_knn[idx * n_samples + j]).enumerate()
                        {
                            ranks_j.insert(neighbor, rank + 1);
                        }

                        // find minimum combined rank of shared neighbors
                        let min_combined_rank = ranks_i
                            .keys()
                            .filter(|&neighbor| ranks_j.contains_key(neighbor))
                            .map(|neighbor| ranks_i[neighbor] + ranks_j[neighbor])
                            .min()
                            .unwrap_or(2 * k)
                            as f32;

                        let preliminary = k as f32 - min_combined_rank / 2.0;
                        let raw_weight = preliminary.max(1e-6);

                        raw_weight / k as f32
                    }
                };

                if weight >= pruning {
                    // Store edge with smaller index first to ensure uniqueness
                    let edge_key = if i < j { (i, j) } else { (j, i) };
                    edges.push((edge_key, weight));
                }
            }

            edges
        })
        .collect::<Vec<_>>()
        .into_iter()
        .fold(FxHashMap::default(), |mut acc, (edge_key, weight)| {
            // Keep the maximum weight if we see the same edge multiple times
            acc.entry(edge_key)
                .and_modify(|existing_weight| {
                    if weight > *existing_weight {
                        *existing_weight = weight;
                    }
                })
                .or_insert(weight);
            acc
        });

    let mut edges = Vec::with_capacity(edge_map.len() * 2);
    let mut weights = Vec::with_capacity(edge_map.len());

    for ((i, j), weight) in edge_map {
        edges.push(i);
        edges.push(j);
        weights.push(weight);
    }

    let end_snn = start_time.elapsed();

    if verbose {
        println!("Transformed kNN into an sNN graph: {:.2?}", end_snn);
    }

    (edges, weights)
}
