use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

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
///
/// ### Fields
///
/// * `offsets` - Stores the offsets for node i's neighbours
/// * `neighbours`- Flat array with neighbour indices
/// * `weights` - Normalised edge weights
#[derive(Debug, Clone)]
pub struct KnnLabPropGraph<T> {
    pub offsets: Vec<usize>,
    pub neighbours: Vec<usize>,
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
                    .and_modify(|w| {
                        *w = match strategy {
                            SymmetryWeightStrategy::Min => w.min(weights[i]),
                            SymmetryWeightStrategy::Max => w.max(weights[i]),
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
