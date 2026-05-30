//! Implementations to generate shared nearest-neighbour graphs from kNN graphs.

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

use crate::prelude::*;

///////////
// Enums //
///////////

/// SNN similarity method
#[derive(Clone, Copy, Default, Debug)]
pub enum SnnSimilarityMethod {
    /// This will calculate the Jaccard similarity as weight
    #[default]
    Intersection,
    /// This will calculate the Rank version as a weight
    Rank,
}

/// Type of shared nearest neighbour graph to create
#[derive(Clone, Copy, Default, Debug)]
pub enum SnnType {
    /// Creates connections between nodes that share nearest neighbours
    #[default]
    FullConnection,
    /// Creates only connections between nodes that are also connected in the
    /// kNN graph
    LimitedConnection,
}

/// Helper function to get the type of sNN similarity
///
/// ### Params
///
/// * `s` - Type of SNN similarity to use
///
/// ### Returns
///
/// Option of the [SnnSimilarityMethod]
pub fn parse_snn_similiarity_method(s: &str) -> Option<SnnSimilarityMethod> {
    match s.to_lowercase().as_str() {
        "jaccard" => Some(SnnSimilarityMethod::Intersection),
        "rank" => Some(SnnSimilarityMethod::Rank),
        _ => None,
    }
}

/// Helper function to get the type of sNN graph construction
///
/// ### Params
///
/// * `s` - Type of sNN graph to create
///
/// ### Returns
///
/// Option of the [SnnType]
pub fn parse_snn_type(s: &str) -> Option<SnnType> {
    match s.to_lowercase().as_str() {
        "full" => Some(SnnType::FullConnection),
        "limited" => Some(SnnType::LimitedConnection),
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
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
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
    verbose: usize,
) -> (Vec<usize>, Vec<f32>) {
    let verbosity = parse_verbosity_level(verbose);

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

    if verbosity.normal_verbosity() {
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
/// * `k` - Number of neighbours in the kNN graph
/// * `n_samples` - Number of samples in the data
/// * `pruning` - Below which Jaccard similarity to prune the edge. In this case
///   the weight is set to `0`.
/// * `method` - Which similarity method to use.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
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
    verbose: usize,
) -> (Vec<usize>, Vec<f32>) {
    let verbosity = parse_verbosity_level(verbose);

    let start_time = Instant::now();

    let edge_map: FxHashMap<(usize, usize), f32> = (0..n_samples)
        .into_par_iter()
        .flat_map(|i| {
            let mut edges = Vec::new();

            // only consider edges to this cell's k nearest neighbors
            for neighbor_idx in 0..k {
                let j = flat_knn[neighbor_idx * n_samples + i];

                // calculate sNN similarity between cell i and its neighbor j
                let weight = match method {
                    SnnSimilarityMethod::Intersection => {
                        // get neighbors of both cells
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

    if verbosity.normal_verbosity() {
        println!("Transformed kNN into an sNN graph: {:.2?}", end_snn);
    }

    (edges, weights)
}

////////////
// Others //
////////////

/// Build an undirected SparseGraph from sNN output.
///
/// ### Params
///
/// * `edges` - Flat edge list as produced by `generate_snn_full` /
///   `generate_snn_limited` (pairs `[u, v, u, v, ...]`).
/// * `weights` - Edge weights, one per pair.
/// * `n_nodes` - Number of nodes in the graph.
pub fn snn_edges_to_sparse_graph(
    edges: &[usize],
    weights: &[f32],
    n_nodes: usize,
) -> SparseGraph<f32> {
    let n_edges = weights.len();

    let mut indptr = vec![0usize; n_nodes + 1];
    for e in 0..n_edges {
        indptr[edges[2 * e] + 1] += 1;
        indptr[edges[2 * e + 1] + 1] += 1;
    }
    for i in 1..=n_nodes {
        indptr[i] += indptr[i - 1];
    }

    let nnz = indptr[n_nodes];
    let mut indices = vec![0usize; nnz];
    let mut data = vec![0.0f32; nnz];
    let mut cursor = indptr.clone();

    for e in 0..n_edges {
        let i = edges[2 * e];
        let j = edges[2 * e + 1];
        let w = weights[e];

        indices[cursor[i]] = j;
        data[cursor[i]] = w;
        cursor[i] += 1;

        indices[cursor[j]] = i;
        data[cursor[j]] = w;
        cursor[j] += 1;
    }

    // sort column indices within each row, keeping data aligned
    for node in 0..n_nodes {
        let start = indptr[node];
        let end = indptr[node + 1];
        let len = end - start;
        if len < 2 {
            continue;
        }

        let mut order: Vec<usize> = (0..len).collect();
        order.sort_unstable_by_key(|&k| indices[start + k]);

        let row_idx: Vec<usize> = order.iter().map(|&k| indices[start + k]).collect();
        let row_data: Vec<f32> = order.iter().map(|&k| data[start + k]).collect();

        indices[start..end].copy_from_slice(&row_idx);
        data[start..end].copy_from_slice(&row_data);
    }

    let adjacency = CompressedSparseData2 {
        data,
        indices: indices.index_cast(),
        indptr: indptr.index_cast(),
        cs_type: CompressedSparseFormat::Csr,
        data_2: None,
        shape: (n_nodes, n_nodes),
    };

    SparseGraph::new(n_nodes, adjacency, false)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    // tiny fixture, 4 nodes, k=2:
    //   node 0 -> [1, 2]
    //   node 1 -> [0, 2]
    //   node 2 -> [0, 1]
    //   node 3 -> [0, 1]
    // column-major: flat_knn[neighbor_idx * n_samples + i]
    fn small_knn() -> (Vec<usize>, usize, usize) {
        let flat_knn = vec![1, 0, 0, 0, 2, 2, 1, 1];
        (flat_knn, 2, 4)
    }

    fn edge_map(edges: &[usize], weights: &[f32]) -> FxHashMap<(usize, usize), f32> {
        edges
            .chunks(2)
            .zip(weights.iter())
            .map(|(p, &w)| ((p[0].min(p[1]), p[0].max(p[1])), w))
            .collect()
    }

    #[test]
    fn snn_full_jaccard_known_weights() {
        let (knn, k, n) = small_knn();
        let (edges, weights) =
            generate_snn_full(&knn, k, n, 0.0, SnnSimilarityMethod::Intersection, 0);
        let m = edge_map(&edges, &weights);

        // all C(4,2)=6 pairs share at least one neighbour
        assert_eq!(m.len(), 6);

        // (0,1): both have neighbour-sets {0,1,2} -> intersection 3, weight 3/(6-3)=1.0
        assert!((m[&(0, 1)] - 1.0).abs() < 1e-6);
        // (0,3): {0,1,2} vs {0,1,3} -> intersection 2, weight 2/(6-2)=0.5
        assert!((m[&(0, 3)] - 0.5).abs() < 1e-6);

        for &w in &weights {
            assert!(w > 0.0 && w <= 1.0);
        }
    }

    #[test]
    fn snn_pruning() {
        let (knn, k, n) = small_knn();
        let (_, w_unpruned) =
            generate_snn_full(&knn, k, n, 0.0, SnnSimilarityMethod::Intersection, 0);
        let (_, w_pruned) =
            generate_snn_full(&knn, k, n, 0.6, SnnSimilarityMethod::Intersection, 0);

        assert!(w_pruned.len() < w_unpruned.len());
        assert!(w_pruned.iter().all(|&w| w >= 0.6));
    }

    #[test]
    fn snn_limited_is_subset_of_full() {
        let (knn, k, n) = small_knn();
        let (fe, fw) = generate_snn_full(&knn, k, n, 0.0, SnnSimilarityMethod::Intersection, 0);
        let (le, lw) = generate_snn_limited(&knn, k, n, 0.0, SnnSimilarityMethod::Intersection, 0);

        let full = edge_map(&fe, &fw);
        let lim = edge_map(&le, &lw);

        assert!(lim.len() <= full.len());
        for (k, w) in &lim {
            assert!(full.contains_key(k));
            assert!((full[k] - w).abs() < 1e-6);
        }
        // 2 and 3 are not kNN of each other -> not in the limited graph
        assert!(!lim.contains_key(&(2, 3)));
    }

    #[test]
    fn snn_rank_weights_valid() {
        let (knn, k, n) = small_knn();
        for f in [generate_snn_full, generate_snn_limited] {
            let (_, weights) = f(&knn, k, n, 0.0, SnnSimilarityMethod::Rank, 0);
            assert!(!weights.is_empty());
            assert!(weights.iter().all(|&w| w > 0.0 && w <= 1.0));
        }
    }

    #[test]
    fn method_parser() {
        assert!(matches!(
            parse_snn_similiarity_method("jaccard"),
            Some(SnnSimilarityMethod::Intersection)
        ));
        assert!(matches!(
            parse_snn_similiarity_method("RANK"),
            Some(SnnSimilarityMethod::Rank)
        ));
        assert!(parse_snn_similiarity_method("nope").is_none());
    }

    // ---- helper tests ----

    #[test]
    fn helper_round_trip() {
        // (0,1) w=0.5, (1,2) w=0.7, (0,2) w=0.3
        let edges = vec![0, 1, 1, 2, 0, 2];
        let weights = vec![0.5, 0.7, 0.3];
        let g = snn_edges_to_sparse_graph(&edges, &weights, 3);

        assert_eq!(g.get_node_number(), 3);
        assert!(!g.is_directed());

        let (n0, w0) = g.get_neighbours(0);
        assert_eq!(n0, &[1, 2]);
        assert_eq!(w0, &[0.5, 0.3]);

        let (n1, w1) = g.get_neighbours(1);
        assert_eq!(n1, &[0, 2]);
        assert_eq!(w1, &[0.5, 0.7]);
    }

    #[test]
    fn helper_indices_sorted() {
        // deliberately scramble the order in which node 0's edges are inserted
        let edges = vec![0, 5, 0, 2, 0, 4, 0, 1, 0, 3];
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let g = snn_edges_to_sparse_graph(&edges, &weights, 6);

        let (nbrs, _) = g.get_neighbours(0);
        assert_eq!(nbrs, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn helper_symmetric() {
        let edges = vec![0, 2, 1, 3, 0, 3];
        let weights = vec![0.4, 0.6, 0.2];
        let g = snn_edges_to_sparse_graph(&edges, &weights, 4);

        for u in 0..4 {
            let (nbrs, ws) = g.get_neighbours(u);
            for (&v, &w) in nbrs.iter().zip(ws.iter()) {
                let (rn, rw) = g.get_neighbours(v);
                let p = rn
                    .iter()
                    .position(|&x| x == u)
                    .expect("reverse edge missing");
                assert!((rw[p] - w).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn helper_total_weight() {
        let edges = vec![0, 1, 1, 2];
        let weights = vec![0.4, 0.6];
        let g = snn_edges_to_sparse_graph(&edges, &weights, 3);

        // undirected -> total_weight halves the doubled adjacency sum, recovers input sum
        assert!((g.total_weight() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn helper_empty() {
        let g = snn_edges_to_sparse_graph(&[], &[], 5);
        assert_eq!(g.get_node_number(), 5);
        for i in 0..5 {
            assert_eq!(g.get_node_degree(i), 0);
        }
    }

    #[test]
    fn helper_end_to_end() {
        let (knn, k, n) = small_knn();
        let (edges, weights) =
            generate_snn_full(&knn, k, n, 0.0, SnnSimilarityMethod::Intersection, 0);
        let g = snn_edges_to_sparse_graph(&edges, &weights, n);

        let total_degree: usize = (0..n).map(|i| g.get_node_degree(i)).sum();
        assert_eq!(total_degree, 2 * weights.len());

        for i in 0..n {
            let (nbrs, _) = g.get_neighbours(i);
            let mut sorted = nbrs.to_vec();
            sorted.sort();
            assert_eq!(nbrs, &sorted[..]);
        }
    }
}
