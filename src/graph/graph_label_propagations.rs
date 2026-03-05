//! Label propagation algorithms over kNN graphs specifically (but with
//! extension to other domains)

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

use crate::assert_same_len;
use crate::prelude::BixverseFloat;

/////////////
// Helpers //
/////////////

/// Which strategy to use for symmetrisation and the weights
#[derive(Debug, Clone, Copy, Default)]
pub enum SymmetryWeightStrategy {
    /// Average weight between two nodes will be taken forward
    #[default]
    Average,
    /// Minimum weight between two nodes will be taken forward
    Min,
    /// Maximum weight between two nodes will be taken forward
    Max,
}

/// Parse the symmetry strategy string
///
/// ### Params
///
/// * `s` - String to parse.
///
/// ### Returns
///
/// The Option of the `SymmetryWeightStrategy` to apply
pub fn parse_symmetry_strategy(s: &str) -> Option<SymmetryWeightStrategy> {
    match s.to_lowercase().as_str() {
        "min" => Some(SymmetryWeightStrategy::Min),
        "max" => Some(SymmetryWeightStrategy::Max),
        "average" | "avg" => Some(SymmetryWeightStrategy::Average),
        _ => None,
    }
}

///////////////////////////
// KNN label propagation //
///////////////////////////

/// Structure to store KNN graphs and do label propagation
#[derive(Debug, Clone)]
pub struct KnnLabPropGraph<T> {
    /// Stores the offsets for node i's neighbours
    pub offsets: Vec<usize>,
    /// Flat array with neighbour indices
    pub neighbours: Vec<usize>,
    /// Normalised edge weights
    pub weights: Vec<T>,
}

impl<T> KnnLabPropGraph<T>
where
    T: BixverseFloat,
{
    //////////////
    // Builders //
    //////////////

    /// Generate the KnnLabelPropGraph
    ///
    /// ### Params
    ///
    /// * `edges` - edge list in form of [node_1, node_2, node_3, ...] which
    ///   indicates alternating pairs (node_1, node_2), etc in terms of edges.
    /// * `n_nodes` - Number of nodes in the graph
    /// * `symmetrise` - Shall the graph be made symmetric.
    ///
    /// ### Returns
    ///
    /// Self with the data stored in the structure.
    pub fn from_edge_list(edges: &[usize], n_nodes: usize, symmetrise: bool) -> Self {
        let mut adj: Vec<Vec<(usize, T)>> = vec![vec![]; n_nodes];

        for chunk in edges.chunks(2) {
            let (u, v) = (chunk[0], chunk[1]);
            adj[u].push((v, T::one()));
            if symmetrise {
                adj[v].push((u, T::one()));
            }
        }

        for neighbours in &mut adj {
            let sum = T::from_usize(neighbours.len()).unwrap();
            for (_, w) in neighbours.iter_mut() {
                *w /= sum;
            }
        }

        Self::build_csr(adj)
    }

    /// Generate KnnLabPropGraph from node pairs
    ///
    /// ### Params
    ///
    /// * `from` - Index of the from node
    /// * `to` - Index of the to node
    /// * `n_nodes` - Number of nodes in the graph
    /// * `symmetrise` - Shall the graph we symmetrised
    ///
    /// ### Returns
    ///
    /// Initialised class
    pub fn from_node_pairs(from: &[usize], to: &[usize], n_nodes: usize, symmetrise: bool) -> Self {
        assert_eq!(from.len(), to.len(), "from and to must have equal length");

        let mut adj: Vec<Vec<(usize, T)>> = vec![vec![]; n_nodes];

        for (&u, &v) in from.iter().zip(to.iter()) {
            adj[u].push((v, T::one()));
            if symmetrise {
                adj[v].push((u, T::one()));
            }
        }

        for neighbours in &mut adj {
            let sum = T::from_usize(neighbours.len()).unwrap();
            for (_, w) in neighbours.iter_mut() {
                *w /= sum;
            }
        }

        Self::build_csr(adj)
    }

    /// Build the KnnLabPropGraph from node pairs (with weights)
    ///
    /// ### Params
    ///
    /// * `from` - Index of the from node
    /// * `to` - Index of the to node
    /// * `weights` - The weights between the nodes
    /// * `n_nodes` - Number of nodes
    /// * `symmetrise` - Symmetrisation strategy
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn from_weighted_node_pairs(
        from: &[usize],
        to: &[usize],
        weights: &[T],
        n_nodes: usize,
        symmetrise: Option<SymmetryWeightStrategy>,
    ) -> Self {
        assert_same_len!(from, to, weights);

        let mut adj: Vec<FxHashMap<usize, T>> = vec![FxHashMap::default(); n_nodes];

        for (i, (&u, &v)) in from.iter().zip(to.iter()).enumerate() {
            adj[u].insert(v, weights[i]);
            if let Some(ref strategy) = symmetrise {
                adj[v]
                    .entry(u)
                    .and_modify(|w: &mut T| {
                        *w = match strategy {
                            SymmetryWeightStrategy::Min => (*w).min(weights[i]),
                            SymmetryWeightStrategy::Max => (*w).max(weights[i]),
                            SymmetryWeightStrategy::Average => {
                                (*w + weights[i]) / T::from_f64(2.0).unwrap()
                            }
                        };
                    })
                    .or_insert(weights[i]);
            }
        }

        let mut adj_vec: Vec<Vec<(usize, T)>> = adj
            .into_iter()
            .map(|map| map.into_iter().collect())
            .collect();

        for neighbours in &mut adj_vec {
            let sum = neighbours
                .iter()
                .map(|(_, w)| *w)
                .fold(T::zero(), |a, b| a + b);
            for (_, w) in neighbours.iter_mut() {
                *w /= sum;
            }
        }

        Self::build_csr(adj_vec)
    }

    /// Generate the KnnLabelPropGraph with weights
    ///
    /// ### Params
    ///
    /// * `edges` - edge list in form of [node_1, node_2, node_3, ...] which
    ///   indicates alternating pairs (node_1, node_2), etc in terms of edges.
    /// * `weights` - the weights between the two nodes in the graph.
    /// * `n_nodes` - Number of nodes in the graph
    /// * `symmetrise` - Which symmetry strategy to use. If None, no
    ///   symmetrisation is performed.
    ///
    /// ### Returns
    ///
    /// Self with the data stored in the structure.
    pub fn from_weighted_edge_list(
        edges: &[usize],
        weights: &[T],
        n_nodes: usize,
        symmetrise: Option<SymmetryWeightStrategy>,
    ) -> Self {
        assert_eq!(
            edges.len() / 2,
            weights.len(),
            "Weight count must match edge count"
        );

        let mut adj: Vec<FxHashMap<usize, T>> = vec![FxHashMap::default(); n_nodes];

        for (i, chunk) in edges.chunks(2).enumerate() {
            let (u, v) = (chunk[0], chunk[1]);
            adj[u].insert(v, weights[i]);
            if let Some(ref strategy) = symmetrise {
                adj[v]
                    .entry(u)
                    .and_modify(|w: &mut T| {
                        // to avoid rust analyser complains
                        *w = match strategy {
                            SymmetryWeightStrategy::Min => (*w).min(weights[i]),
                            SymmetryWeightStrategy::Max => (*w).max(weights[i]),
                            SymmetryWeightStrategy::Average => {
                                (*w + weights[i]) / T::from_f64(2.0).unwrap()
                            }
                        };
                    })
                    .or_insert(weights[i]);
            }
        }

        let mut adj_vec: Vec<Vec<(usize, T)>> = adj
            .into_iter()
            .map(|map| map.into_iter().collect())
            .collect();

        for neighbours in &mut adj_vec {
            let sum = neighbours
                .iter()
                .map(|(_, w)| *w)
                .fold(T::zero(), |a, b| a + b);
            for (_, w) in neighbours.iter_mut() {
                *w /= sum;
            }
        }

        Self::build_csr(adj_vec)
    }

    /// Label spreading algorithm
    ///
    /// Spreads labels (one-hot encoded categorical data) over the graph. The
    /// input needs to be of structure:
    ///
    /// class 1 -> `[1.0, 0.0, 0.0, 0.0]`
    ///
    /// class 2 -> `[0.0, 1.0, 0.0, 0.0]`
    ///
    /// unlabelled -> `[0.0, 0.0, 0.0, 0.0]`
    ///
    /// Labelled nodes are anchored to their original labels and act as sources,
    /// while unlabelled nodes receive purely propagated distributions.
    ///
    /// ### Params
    ///
    /// * `labels` - One-hot encoded group membership. All zeroes == unlabelled.
    /// * `mask` - Boolean slice where `true` indicates an unlabelled node.
    /// * `alpha` - Anchoring strength for labelled nodes, typically 0.9 to 0.95.
    ///   Higher values anchor more strongly to the original label; lower values
    ///   allow more influence from neighbours.
    /// * `iterations` - Maximum number of spreading iterations.
    /// * `tolerance` - Convergence threshold. Stops early if the maximum change
    ///   across all nodes and classes falls below this value.
    /// * `max_hops` - If `Some(k)`, restricts spreading to nodes within `k` hops
    ///   of any labelled node. Nodes beyond this limit are never updated and
    ///   remain as all-zeroes. If `None`, spreading is unrestricted.
    ///
    /// ### Returns
    ///
    /// A `Vec<Vec<T>>` of length `n_nodes` where each entry is the probability
    /// distribution over classes for that node.
    pub fn label_spreading(
        &self,
        labels: &[Vec<T>],
        mask: &[bool],
        alpha: T,
        iterations: usize,
        tolerance: T,
        max_hops: Option<usize>,
    ) -> Vec<Vec<T>> {
        let n = labels.len();
        let num_classes = labels[0].len();
        let mut y = labels.to_vec();
        let mut y_new = vec![vec![T::zero(); num_classes]; n];

        let distances = max_hops.map(|_| self.compute_hop_distances(mask));

        for _ in 0..iterations {
            y_new.par_iter_mut().enumerate().for_each(|(node, y_dist)| {
                if let (Some(h), Some(dists)) = (max_hops, &distances)
                    && dists[node] > h
                {
                    return;
                }

                let start = self.offsets[node];
                let end = self.offsets[node + 1];
                y_dist.fill(T::zero());
                for i in start..end {
                    let neighbor_dist = &y[self.neighbours[i]];
                    for c in 0..num_classes {
                        y_dist[c] += self.weights[i] * neighbor_dist[c];
                    }
                }
            });

            let max_change = y
                .par_iter_mut()
                .enumerate()
                .map(|(i, y_dist)| {
                    if let (Some(h), Some(dists)) = (max_hops, &distances)
                        && dists[i] > h
                    {
                        return T::zero();
                    }

                    let mut max_diff = T::zero();
                    if mask[i] {
                        for c in 0..num_classes {
                            max_diff = max_diff.max((y_new[i][c] - y_dist[c]).abs());
                            y_dist[c] = y_new[i][c];
                        }
                    } else {
                        for c in 0..num_classes {
                            let new_val = alpha * labels[i][c] + (T::one() - alpha) * y_new[i][c];
                            max_diff = max_diff.max((new_val - y_dist[c]).abs());
                            y_dist[c] = new_val;
                        }
                    }
                    max_diff
                })
                .reduce(|| T::zero(), T::max);

            if max_change < tolerance {
                break;
            }
        }

        y
    }

    /// Internal helper to generate the CSR representation
    ///
    /// ### Params
    ///
    /// * `build_csr` - The adjacency of the graph
    ///
    /// ### Returns
    ///
    /// Self.
    fn build_csr(adj: Vec<Vec<(usize, T)>>) -> Self {
        let mut offsets = vec![0];
        let mut neighbours = Vec::new();
        let mut weights = Vec::new();

        for node_neighbours in adj {
            for (neighbour, weight) in node_neighbours {
                neighbours.push(neighbour);
                weights.push(weight);
            }
            offsets.push(neighbours.len());
        }

        Self {
            offsets,
            neighbours,
            weights,
        }
    }

    /// Compute hop distances
    ///
    /// ### Params
    ///
    /// * `mask` -
    ///
    /// ### Return
    ///
    ///
    fn compute_hop_distances(&self, mask: &[bool]) -> Vec<usize> {
        let n = mask.len();
        let mut distances = vec![usize::MAX; n];
        let mut queue = VecDeque::new();

        for (i, &is_unlabelled) in mask.iter().enumerate() {
            if !is_unlabelled {
                distances[i] = 0;
                queue.push_back(i);
            }
        }

        while let Some(node) = queue.pop_front() {
            let start = self.offsets[node];
            let end = self.offsets[node + 1];

            for i in start..end {
                let neighbour = self.neighbours[i];
                if distances[neighbour] == usize::MAX {
                    distances[neighbour] = distances[node] + 1;
                    queue.push_back(neighbour);
                }
            }
        }

        distances
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    type Graph = KnnLabPropGraph<f64>;

    // 0 -- 1 -- 2
    fn simple_chain() -> Graph {
        let edges = vec![0, 1, 1, 2];
        Graph::from_edge_list(&edges, 3, true)
    }

    #[test]
    fn test_csr_structure_symmetric() {
        let g = simple_chain();
        // Node 0: [1], Node 1: [0, 2], Node 2: [1]
        assert_eq!(g.offsets, vec![0, 1, 3, 4]);
        assert_eq!(g.neighbours.len(), 4);
    }

    #[test]
    fn test_weights_normalised() {
        let g = simple_chain();
        // Node 1 has two neighbours, each weight should be 0.5
        let start = g.offsets[1];
        let end = g.offsets[2];
        let sum: f64 = g.weights[start..end].iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_symmetry_average() {
        // Edge 0->1 weight 0.8, Edge 1->0 weight 0.4 => symmetric weight 0.6
        let edges = vec![0, 1, 1, 0];
        let weights = vec![0.8_f64, 0.4];
        let g = Graph::from_weighted_edge_list(
            &edges,
            &weights,
            2,
            Some(SymmetryWeightStrategy::Average),
        );
        // Both nodes have a single neighbour so weight == 1.0 after normalisation,
        // but the raw value before normalisation should have been averaged.
        // With a single edge per node, normalised weight is always 1.0.
        assert!((g.weights[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_symmetry_min() {
        let edges = vec![0, 1, 1, 0];
        let weights = vec![0.8_f64, 0.4];
        let g =
            Graph::from_weighted_edge_list(&edges, &weights, 2, Some(SymmetryWeightStrategy::Min));
        assert!((g.weights[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_label_spreading_labelled_nodes_anchored() {
        let g = simple_chain();
        let labels = vec![vec![1.0_f64, 0.0], vec![0.0, 0.0], vec![0.0, 1.0]];
        let mask = vec![false, true, false];
        let result = g.label_spreading(&labels, &mask, 0.9, 100, 1e-6, None);

        // Labelled nodes are soft-anchored, not hard-clamped, so just verify dominance
        assert!(
            result[0][0] > result[0][1],
            "node 0 should still favour class 0"
        );
        assert!(
            result[2][1] > result[2][0],
            "node 2 should still favour class 1"
        );
    }

    #[test]
    fn test_label_spreading_unlabelled_receives_distribution() {
        let g = simple_chain();
        let labels = vec![vec![1.0_f64, 0.0], vec![0.0, 0.0], vec![0.0, 1.0]];
        let mask = vec![false, true, false];
        let result = g.label_spreading(&labels, &mask, 0.9, 100, 1e-6, None);

        // Node 1 sits between class 0 and class 1 so should end up roughly 0.5/0.5
        assert!((result[1][0] - 0.5).abs() < 0.05);
        assert!((result[1][1] - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_max_hops_restricts_propagation() {
        // 0 -- 1 -- 2 -- 3 -- 4
        // Node 0 labelled, rest unlabelled, max_hops = 2
        let edges = vec![0, 1, 1, 2, 2, 3, 3, 4];
        let g = Graph::from_edge_list(&edges, 5, true);
        let labels = vec![
            vec![1.0_f64, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];
        let mask = vec![false, true, true, true, true];
        let result = g.label_spreading(&labels, &mask, 0.9, 100, 1e-6, Some(2));

        // Node 4 is 4 hops away, should remain all-zero
        assert!((result[4][0]).abs() < 1e-9);
        assert!((result[4][1]).abs() < 1e-9);
    }

    #[test]
    fn test_hop_distances() {
        let g = simple_chain();
        let mask = vec![false, true, false];
        let dists = g.compute_hop_distances(&mask);
        assert_eq!(dists[0], 0);
        assert_eq!(dists[1], 1);
        assert_eq!(dists[2], 0);
    }

    #[test]
    fn test_parse_symmetry_strategy() {
        assert!(matches!(
            parse_symmetry_strategy("min"),
            Some(SymmetryWeightStrategy::Min)
        ));
        assert!(matches!(
            parse_symmetry_strategy("MAX"),
            Some(SymmetryWeightStrategy::Max)
        ));
        assert!(matches!(
            parse_symmetry_strategy("avg"),
            Some(SymmetryWeightStrategy::Average)
        ));
        assert!(parse_symmetry_strategy("nonsense").is_none());
    }

    #[test]
    fn test_node_pairs_matches_edge_list() {
        let from = vec![0, 1];
        let to = vec![1, 2];
        let g = Graph::from_node_pairs(&from, &to, 3, true);
        assert_eq!(g.offsets, vec![0, 1, 3, 4]);
        assert_eq!(g.neighbours.len(), 4);
    }

    #[test]
    fn test_node_pairs_weights_normalised() {
        let from = vec![0, 1];
        let to = vec![1, 2];
        let g = Graph::from_node_pairs(&from, &to, 3, true);
        let start = g.offsets[1];
        let end = g.offsets[2];
        let sum: f64 = g.weights[start..end].iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_node_pairs_symmetry_average() {
        let from = vec![0];
        let to = vec![1];
        let weights = vec![0.8_f64];
        let g = Graph::from_weighted_node_pairs(
            &from,
            &to,
            &weights,
            2,
            Some(SymmetryWeightStrategy::Average),
        );
        assert!((g.weights[0] - 1.0).abs() < 1e-9);
        assert!((g.weights[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_node_pairs_matches_weighted_edge_list() {
        let from = vec![0, 1];
        let to = vec![1, 2];
        let weights = vec![0.6_f64, 0.4];

        let g_pairs = Graph::from_weighted_node_pairs(
            &from,
            &to,
            &weights,
            3,
            Some(SymmetryWeightStrategy::Min),
        );
        let edges = vec![0, 1, 1, 2];
        let g_edge =
            Graph::from_weighted_edge_list(&edges, &weights, 3, Some(SymmetryWeightStrategy::Min));

        assert_eq!(g_pairs.offsets, g_edge.offsets);
        assert_eq!(g_pairs.neighbours, g_edge.neighbours);
        for (a, b) in g_pairs.weights.iter().zip(g_edge.weights.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }
}
