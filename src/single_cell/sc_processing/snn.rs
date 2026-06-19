//! Implementations to generate shared nearest-neighbour graphs from kNN graphs.

use rayon::prelude::*;
use rustc_hash::FxHashMap;
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

    let start_time = Instant::now();

    let knn_rm: Vec<usize> = {
        let mut v = vec![0usize; n_samples * k];
        for i in 0..n_samples {
            for nb in 0..k {
                v[i * k + nb] = flat_knn[nb * n_samples + i];
            }
        }
        v
    };

    let mut reverse_mappings: Vec<Vec<(usize, usize)>> =
        (0..n_samples).map(|_| Vec::with_capacity(k + 1)).collect();

    for i in 0..n_samples {
        reverse_mappings[i].push((i, 0));
        for nb in 0..k {
            let neighbor = flat_knn[nb * n_samples + i];
            reverse_mappings[neighbor].push((i, nb + 1));
        }
    }
    for nb in 0..k {
        for i in 0..n_samples {
            let neighbor = flat_knn[nb * n_samples + i];
            reverse_mappings[neighbor].push((i, nb + 1));
        }
    }

    let results: Vec<(usize, usize, f32)> = (0..n_samples)
        .into_par_iter()
        .map_init(
            || (vec![0.0f32; n_samples], Vec::<usize>::with_capacity(64)),
            |(scores, added), j| {
                added.clear();

                for i in 0..=k {
                    let cur_neighbor = if i == 0 { j } else { knn_rm[j * k + (i - 1)] };

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

                let out: Vec<(usize, usize, f32)> = added
                    .iter()
                    .filter_map(|&othernode| {
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
                        (weight >= pruning).then_some((j, othernode, weight))
                    })
                    .collect();

                for &othernode in added.iter() {
                    scores[othernode] = 0.0;
                }

                out
            },
        )
        .flatten()
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

    let knn_rm: Vec<usize> = {
        let mut v = vec![0usize; n_samples * k];
        for r in 0..k {
            let col_offset = r * n_samples;
            for c in 0..n_samples {
                v[c * k + r] = flat_knn[col_offset + c];
            }
        }
        v
    };

    let attached: Vec<Vec<(usize, u32)>> = (0..n_samples)
        .into_par_iter()
        .map(|c| {
            let mut v: Vec<(usize, u32)> = Vec::with_capacity(k + 1);
            v.push((c, 0));
            let row_start = c * k;
            for r in 0..k {
                v.push((knn_rm[row_start + r], (r + 1) as u32));
            }
            v.sort_unstable_by_key(|&(node, _)| node);
            v
        })
        .collect();

    let edge_map: FxHashMap<(usize, usize), f32> = (0..n_samples)
        .into_par_iter()
        .fold(FxHashMap::<(usize, usize), f32>::default, |mut acc, i| {
            let ai = &attached[i];
            let row_start = i * k;

            for r in 0..k {
                let j = knn_rm[row_start + r];
                let aj = &attached[j];

                let weight = match method {
                    SnnSimilarityMethod::Intersection => {
                        let (mut pi, mut pj) = (0usize, 0usize);
                        let mut count: u32 = 0;
                        while pi < ai.len() && pj < aj.len() {
                            let ni = ai[pi].0;
                            let nj = aj[pj].0;
                            if ni == nj {
                                count += 1;
                                pi += 1;
                                pj += 1;
                            } else if ni < nj {
                                pi += 1;
                            } else {
                                pj += 1;
                            }
                        }
                        let c = count as f32;
                        c / (2.0 * (k as f32 + 1.0) - c)
                    }
                    SnnSimilarityMethod::Rank => {
                        let (mut pi, mut pj) = (0usize, 0usize);
                        let mut min_combined: u32 = 2 * (k as u32);
                        while pi < ai.len() && pj < aj.len() {
                            let (ni, ri) = ai[pi];
                            let (nj, rj) = aj[pj];
                            if ni == nj {
                                let combined = ri + rj;
                                if combined < min_combined {
                                    min_combined = combined;
                                }
                                pi += 1;
                                pj += 1;
                            } else if ni < nj {
                                pi += 1;
                            } else {
                                pj += 1;
                            }
                        }
                        let preliminary = k as f32 - min_combined as f32 / 2.0;
                        preliminary.max(1e-6) / k as f32
                    }
                };

                if weight >= pruning {
                    let edge_key = if i < j { (i, j) } else { (j, i) };
                    acc.entry(edge_key)
                        .and_modify(|w| {
                            if weight > *w {
                                *w = weight
                            }
                        })
                        .or_insert(weight);
                }
            }
            acc
        })
        .reduce(FxHashMap::<(usize, usize), f32>::default, |mut a, mut b| {
            if a.len() < b.len() {
                std::mem::swap(&mut a, &mut b);
            }
            for (key, v) in b {
                a.entry(key)
                    .and_modify(|w| {
                        if v > *w {
                            *w = v
                        }
                    })
                    .or_insert(v);
            }
            a
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
