use petgraph::Graph;
use petgraph::prelude::*;
use petgraph::visit::NodeIndexable;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::prelude::*;

/////////////
// Helpers //
/////////////

const DEFAULT_TOLERANCE: f64 = 1e-6;

/// Structure for Page Rank Memory
///
/// Allows for faster, better usage of memory
///
/// ### Fields
///
/// * `ranks` - The old ranks
/// * `new_ranks` - The new ranks
#[derive(Debug)]
pub struct PageRankWorkingMemory<T> {
    ranks: Vec<T>,
    new_ranks: Vec<T>,
}

impl<T> PageRankWorkingMemory<T>
where
    T: BixverseFloat,
{
    /// Initialise the structure
    ///
    /// ### Returns
    ///
    /// Initialised `PageRankWorkingMemory` structure
    pub fn new() -> Self {
        Self {
            ranks: Vec::new(),
            new_ranks: Vec::new(),
        }
    }

    /// Ensure that the capacity is correct to avoid panics
    ///
    /// ### Params
    ///
    /// * `node_count` - The new node count.
    fn ensure_capacity(&mut self, node_count: usize) {
        if self.ranks.len() < node_count {
            self.ranks.resize(node_count, T::zero());
            self.new_ranks.resize(node_count, T::zero());
        }
    }
}

impl<T> Default for PageRankWorkingMemory<T>
where
    T: BixverseFloat,
{
    fn default() -> Self {
        Self {
            ranks: Vec::new(),
            new_ranks: Vec::new(),
        }
    }
}

/// Precomputed graph structure for efficient PageRank computation
///
/// ### Fields
///
/// * `node_count` - The total number of nodes in the graph
/// * `in_edges_flat` - Flattened incoming edges: `[node0_in_edges..., node1_in_edges..., ...]`
/// * `in_edges_offsets` - Offsets into the `in_edges_flat` for each node
/// * `out_degrees` - The out degree for each node.
#[derive(Clone)]
pub struct PageRankGraph<T> {
    node_count: usize,
    in_edges_flat: Vec<usize>,
    in_edge_weights_flat: Vec<T>,
    in_edges_offsets: Vec<usize>,
    out_weight_sums: Vec<T>,
}

#[allow(dead_code)]
impl<T> PageRankGraph<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    /// Generate the structure from a given petgraph.
    ///
    /// ### Params
    ///
    /// * `graph` The PetGraph from which to generate the structure.
    ///
    /// ### Returns
    ///
    /// Initialised `PageRankGraph` structure
    pub fn from_petgraph(graph: Graph<&str, T>) -> Self {
        let node_count = graph.node_count();

        // Build adjacency structure more efficiently
        let mut out_edges: Vec<Vec<(usize, T)>> = vec![Vec::new(); node_count];
        let mut in_edges: Vec<Vec<(usize, T)>> = vec![Vec::new(); node_count];

        // Single pass through edges and clippy being dumb

        for i in 0..node_count {
            let node_id = graph.from_index(i);
            for edge in graph.edges(node_id) {
                let target_idx = graph.to_index(edge.target());
                let weight = *edge.weight();
                out_edges[i].push((target_idx, weight));
                in_edges[target_idx].push((i, weight));
            }
        }

        // Flatten in_edges for better cache locality
        let mut in_edges_flat = Vec::new();
        let mut in_edge_weights_flat = Vec::new();
        let mut in_edges_offsets = Vec::with_capacity(node_count + 1);
        in_edges_offsets.push(0);

        for node_in_edges in &in_edges {
            for &(node_idx, weight) in node_in_edges {
                in_edges_flat.push(node_idx);
                in_edge_weights_flat.push(weight);
            }
            in_edges_offsets.push(in_edges_flat.len());
        }

        // Calculate sum of outgoing weights for each node
        let out_weight_sums: Vec<T> = out_edges
            .iter()
            .map(|edges| edges.iter().map(|(_, weight)| *weight).sum())
            .collect();

        Self {
            node_count,
            in_edges_flat,
            in_edge_weights_flat,
            in_edges_offsets,
            out_weight_sums,
        }
    }

    /// Generate the structure directly from node names and edge lists
    ///
    /// ### Params
    ///
    /// * `nodes` - Slice of the node names
    /// * `from` - Slice of the names of the from nodes
    /// * `to` - Slice of the names of the to nodes
    /// * `undirected` - Whether to create bidirectional edges
    ///
    /// ### Returns
    ///
    /// Initialised `PageRankGraph` structure
    pub fn from_strings(
        nodes: &[String],
        from: &[String],
        to: &[String],
        weights: Option<&[T]>,
        undirected: bool,
    ) -> Self {
        assert_same_len!(from, to);

        if let Some(weights) = weights {
            assert_same_len!(from, weights);
        }

        let node_count = nodes.len();

        // Create mapping from node names to indices
        let mut name_to_idx = FxHashMap::default();
        for (idx, name) in nodes.iter().enumerate() {
            name_to_idx.insert(name, idx);
        }

        // Build adjacency lists with weights
        let mut out_edges: Vec<Vec<(usize, T)>> = vec![Vec::new(); node_count];
        let mut in_edges: Vec<Vec<(usize, T)>> = vec![Vec::new(); node_count];

        for (i, (from_name, to_name)) in from.iter().zip(to.iter()).enumerate() {
            let from_idx = *name_to_idx.get(from_name).unwrap();
            let to_idx = *name_to_idx.get(to_name).unwrap();

            let weight = weights.map(|w| w[i]).unwrap_or(T::zero());

            out_edges[from_idx].push((to_idx, weight));
            in_edges[to_idx].push((from_idx, weight));

            if undirected {
                out_edges[to_idx].push((from_idx, weight));
                in_edges[from_idx].push((to_idx, weight));
            }
        }

        // Flatten in_edges for better cache locality
        let mut in_edges_flat = Vec::new();
        let mut in_edge_weights_flat = Vec::new();
        let mut in_edges_offsets = Vec::with_capacity(node_count + 1);
        in_edges_offsets.push(0);

        for node_in_edges in &in_edges {
            for &(node_idx, weight) in node_in_edges {
                in_edges_flat.push(node_idx);
                in_edge_weights_flat.push(weight);
            }
            in_edges_offsets.push(in_edges_flat.len());
        }

        // Calculate sum of outgoing weights for each node
        let out_weight_sums: Vec<T> = out_edges
            .iter()
            .map(|edges| edges.iter().map(|(_, weight)| *weight).sum())
            .collect();

        Self {
            node_count,
            in_edges_flat,
            in_edge_weights_flat,
            in_edges_offsets,
            out_weight_sums,
        }
    }

    /// Get incoming edges for a node
    ///
    /// Inline function to hopefully optimise further the compilation of the
    /// program
    ///
    /// ### Params
    ///
    /// * `node` - Get the in_edges for a given node index.
    ///
    /// ### Return
    ///
    /// Returns a slice of in_edges
    #[inline]
    fn in_edges(&self, node: usize) -> (&[usize], &[T]) {
        let start = self.in_edges_offsets[node];
        let end = self.in_edges_offsets[node + 1];
        (
            &self.in_edges_flat[start..end],
            &self.in_edge_weights_flat[start..end],
        )
    }
}

//////////////
// PageRank //
//////////////

/// Parallel personalised PageRank algorithm.
///
/// ### Params
///
/// * `graph` - The PetGraph on which to run the personalised page-rank.
/// * `damping_factor` - The dampening factor parameter, i.e., the probability
///   of resetting.
/// * `personalization_vector` - The vector of probabilities for the reset,
///   making this the personalised page rank.
/// * `nb_iter` - Maximum number of iterations for the personalised page rank.
/// * `tolerance` - Optional tolerance for the algorithm. If not provided, it will
///   default to `1e-6`.
///
/// ### Returns
///
/// The (normalised) personalised page rank scores.
pub fn personalised_page_rank<T>(
    graph: Graph<&str, T>,
    damping_factor: T,
    personalisation_vector: &[T],
    nb_iter: usize,
    tol: Option<T>,
) -> Vec<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    let node_count = graph.node_count();
    if node_count == 0 {
        return vec![];
    }

    // Validate inputs (same as before)
    assert!(
        T::zero() <= damping_factor && damping_factor <= T::one(),
        "Damping factor should be between 0 and 1."
    );
    assert_eq!(
        personalisation_vector.len(),
        node_count,
        "Personalisation vector length must match node count."
    );

    let tolerance = tol.unwrap_or(T::from_f64(DEFAULT_TOLERANCE).unwrap());

    let mut out_edges: Vec<Vec<(usize, T)>> = vec![Vec::new(); node_count];
    let mut in_edges: Vec<Vec<(usize, T)>> = vec![Vec::new(); node_count];

    // build adjacency lists with weights
    for (i, out_edge_vec) in out_edges.iter_mut().enumerate().take(node_count) {
        let node_id = graph.from_index(i);
        for edge in graph.edges(node_id) {
            let target_idx = graph.to_index(edge.target());
            let weight = *edge.weight();
            out_edge_vec.push((target_idx, weight));
            in_edges[target_idx].push((i, weight));
        }
    }

    let out_weight_sums: Vec<T> = out_edges
        .iter()
        .map(|edges| edges.iter().map(|(_, weight)| *weight).sum())
        .collect();

    let mut ranks: Vec<T> = personalisation_vector.to_vec();
    let teleport_factor = T::one() - damping_factor;

    for _ in 0..nb_iter {
        let new_ranks: Vec<T> = (0..node_count)
            .into_par_iter()
            .map(|v| {
                let teleport_prob = teleport_factor * personalisation_vector[v];

                let link_prob = in_edges[v]
                    .iter()
                    .map(|&(w, edge_weight)| {
                        if out_weight_sums[w] > T::zero() {
                            damping_factor * ranks[w] * edge_weight / out_weight_sums[w]
                        } else {
                            damping_factor * ranks[w] * personalisation_vector[v]
                        }
                    })
                    .sum::<T>();

                teleport_prob + link_prob
            })
            .collect();

        let squared_norm_2 = new_ranks
            .par_iter()
            .zip(&ranks)
            .map(|(new, old)| (*new - *old) * (*new - *old))
            .sum::<T>();

        ranks = new_ranks;

        if squared_norm_2 <= tolerance {
            break;
        }
    }

    let sum: T = ranks.iter().copied().sum();
    ranks.iter_mut().for_each(|x| *x /= sum);

    ranks
}

/// Optimised PageRank with pre-allocated working memory
///
/// This is a highly optimised version of the personalised page rank to be used
/// for rapid permutations.
///
/// ### Params
///
/// * `graph` - The `PageRankGraph` structure with pre-computed values for
///   fast calculations
/// * `damping_factor` - The dampening factor parameter, i.e., the probability
///   of resetting.
/// * `personalization_vector` - The vector of probabilities for the reset,
///   making this the personalised page rank.
/// * `nb_iter` - Maximum number of iterations for the personalised page rank.
/// * `tolerance` - Tolerance of the algorithm.
/// * `working_memory` - The `PageRankWorkingMemory` structure to store the old
///   and new ranks
///
/// ### Returns
///
/// The (normalised) personalised page rank scores.
pub fn personalised_page_rank_optimised<T>(
    graph: &PageRankGraph<T>,
    damping_factor: T,
    personalisation_vector: &[T],
    nb_iter: usize,
    tolerance: T,
    working_memory: &mut PageRankWorkingMemory<T>,
) -> Vec<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    let node_count = graph.node_count;

    // reuse pre-allocated vectors
    working_memory.ensure_capacity(node_count);
    let ranks = &mut working_memory.ranks;
    let new_ranks = &mut working_memory.new_ranks;

    // initialise ranks
    ranks[..node_count].copy_from_slice(personalisation_vector);

    let teleport_factor = T::one() - damping_factor;

    for _ in 0..nb_iter {
        // compute new ranks
        new_ranks[..node_count]
            .par_iter_mut()
            .enumerate()
            .for_each(|(v, new_rank)| {
                let teleport_prob = teleport_factor * personalisation_vector[v];

                let (in_nodes, in_weights) = graph.in_edges(v);
                let link_prob: T = in_nodes
                    .iter()
                    .zip(in_weights.iter())
                    .map(|(&w, &edge_weight)| {
                        if graph.out_weight_sums[w] > T::zero() {
                            damping_factor * ranks[w] * edge_weight / graph.out_weight_sums[w]
                        } else {
                            damping_factor * ranks[w] * personalisation_vector[v]
                        }
                    })
                    .sum();

                *new_rank = teleport_prob + link_prob;
            });

        // Check convergence
        let squared_norm_2: T = new_ranks[..node_count]
            .par_iter()
            .zip(&ranks[..node_count])
            .map(|(new, old)| {
                let diff = *new - *old;
                diff * diff
            })
            .sum();

        // Swap vectors (no allocation)
        std::mem::swap(ranks, new_ranks);

        if squared_norm_2 <= tolerance {
            break;
        }
    }

    // Normalize (make sure that sum == 1)
    let sum: T = ranks[..node_count].iter().cloned().sum();
    if sum > T::zero() {
        ranks[..node_count].iter_mut().for_each(|x| *x /= sum);
    }

    ranks[..node_count].to_vec()
}

////////////////////////////////////////
// Constrained personalised page rank //
////////////////////////////////////////

/// Constrained parallel personalised PageRank algorithm
///
/// ### Params
///
/// * `graph` - The PetGraph with NodeData and EdgeData
/// * `damping_factor` - The dampening factor parameter, i.e., the probability
///   of resetting.
/// * `personalization_vector` - The vector of probabilities for the reset,
///   making this the personalised page rank.
/// * `nb_iter` - Maximum number of iterations for the personalised page rank.
/// * `tolerance` - Optional tolerance for the algorithm. If not provided, it
///   will default to `1e-6`.
/// * `sink_node_types` - Optional HashSet of node types that act as sinks
///   (force reset of the surfer)
/// * `constrained_edge_types` - Optional HashSet of edge types that force reset
///   after traversal of that edge.
///
///
/// ### Returns
///
/// The normalised personalised PageRank scores
pub fn constrained_personalised_page_rank<T>(
    graph: &Graph<NodeData, EdgeData<T>>,
    damping_factor: T,
    personalisation_vector: &[T],
    nb_iter: usize,
    tol: Option<T>,
    sink_node_types: Option<&FxHashSet<String>>,
    constrained_edge_types: Option<&FxHashSet<String>>,
) -> Vec<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    let node_count = graph.node_count();
    if node_count == 0 {
        return vec![];
    }

    // further assertions
    assert!(
        T::zero() <= damping_factor && damping_factor <= T::one(),
        "Damping factor should be between 0 and 1."
    );
    assert_eq!(
        personalisation_vector.len(),
        node_count,
        "Personalization vector length must match node count."
    );

    let tolerance = tol.unwrap_or(T::from_f64(DEFAULT_TOLERANCE).unwrap());
    let binding = FxHashSet::default();
    let sink_types = sink_node_types.unwrap_or(&binding);
    let constrained_types = constrained_edge_types.unwrap_or(&binding);

    // build transition structure with constraints and weights
    let mut out_edges: Vec<Vec<(usize, T, bool)>> = vec![Vec::new(); node_count];
    let mut in_edges: Vec<Vec<(usize, T, bool)>> = vec![Vec::new(); node_count];

    for node_idx in graph.node_indices() {
        let node_idx_usize = graph.to_index(node_idx);
        let node_data = &graph[node_idx];

        // check if this node is a sink - if so, it has no valid outgoing edges
        if sink_types.contains(node_data.node_type) {
            continue;
        }

        // add all outgoing edges with their weights and sink edge flags
        for edge in graph.edges(node_idx) {
            let edge_data = edge.weight();
            let target_idx = graph.to_index(edge.target());
            let is_sink_edge = constrained_types.contains(edge_data.edge_type);

            out_edges[node_idx_usize].push((target_idx, *edge_data.weight, is_sink_edge));
            in_edges[target_idx].push((node_idx_usize, *edge_data.weight, is_sink_edge));
        }
    }

    // calculate out-degrees (sum of weights of ALL edges, including sink edges)
    let out_degrees: Vec<T> = out_edges
        .iter()
        .map(|edges| {
            let total_weight: T = edges.iter().map(|(_, weight, _)| *weight).sum();
            total_weight
        })
        .collect();

    // track mass that can flow forward (non-sink mass) vs absorbed mass (sink mass)
    let mut flowable_ranks: Vec<T> = personalisation_vector.to_vec();
    let mut absorbed_ranks: Vec<T> = vec![T::zero(); node_count];
    let teleport_factor = T::one() - damping_factor;

    for _ in 0..nb_iter {
        // Calculate new flowable and absorbed mass
        let (new_flowable, new_absorbed): (Vec<T>, Vec<T>) = (0..node_count)
            .into_par_iter()
            .map(|v| {
                let teleport_mass = teleport_factor * personalisation_vector[v];
                let mut flowable = teleport_mass;
                let mut absorbed = T::zero();

                for &(w, edge_weight, is_sink_edge) in &in_edges[v] {
                    if out_degrees[w] > T::zero() {
                        let flow =
                            damping_factor * flowable_ranks[w] * edge_weight / out_degrees[w];

                        if is_sink_edge {
                            absorbed += flow;
                        } else {
                            flowable += flow;
                        }
                    } else {
                        // Source node w has no outgoing edges (sink node)
                        flowable += damping_factor * flowable_ranks[w] * personalisation_vector[v];
                    }
                }

                (flowable, absorbed)
            })
            .unzip();

        // Total absorbed mass gets teleported back
        let total_absorbed_mass: T = new_absorbed.iter().copied().sum();

        let final_flowable: Vec<T> = new_flowable
            .into_iter()
            .enumerate()
            .map(|(v, flowable)| flowable + total_absorbed_mass * personalisation_vector[v])
            .collect();

        // Total ranks = flowable + absorbed
        let total_ranks: Vec<T> = final_flowable
            .iter()
            .zip(&new_absorbed)
            .map(|(f, a)| *f + *a)
            .collect();

        // Check for convergence using total ranks
        let squared_norm_2 = total_ranks
            .par_iter()
            .zip(flowable_ranks.par_iter().zip(&absorbed_ranks))
            .map(|(new_total, (old_flow, old_abs))| {
                let old_total = *old_flow + *old_abs;
                (*new_total - old_total) * (*new_total - old_total)
            })
            .sum::<T>();

        flowable_ranks = final_flowable;
        absorbed_ranks = new_absorbed;

        if squared_norm_2 <= tolerance {
            break;
        }
    }

    // final ranks = flowable + absorbed
    let mut ranks: Vec<T> = flowable_ranks
        .iter()
        .zip(&absorbed_ranks)
        .map(|(f, a)| *f + *a)
        .collect();

    // normalise
    let sum: T = ranks.iter().copied().sum();
    if sum > T::zero() {
        ranks.iter_mut().for_each(|x| *x /= sum);
    }

    ranks
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_rank_star() {
        let mut graph = Graph::<&str, f64>::new();
        let n0 = graph.add_node("0");
        let n1 = graph.add_node("1");
        let n2 = graph.add_node("2");

        // 0 links to 1 and 2
        graph.add_edge(n0, n1, 1.0);
        graph.add_edge(n0, n2, 1.0);

        // Uniform personalization
        let p_vec = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

        let ranks = personalised_page_rank(graph, 0.85, &p_vec, 100, None);

        assert_eq!(ranks.len(), 3);
        // Symmetry: 1 and 2 must have exactly the same rank
        assert!((ranks[1] - ranks[2]).abs() < 1e-6);
        // Mass conservation
        assert!((ranks.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_optimised_page_rank_matches() {
        let nodes = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let from = vec!["A".to_string(), "A".to_string()];
        let to = vec!["B".to_string(), "C".to_string()];
        let weights = vec![1.0, 1.0];

        let pr_graph = PageRankGraph::from_strings(&nodes, &from, &to, Some(&weights), false);

        let p_vec: Vec<f64> = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let mut memory = PageRankWorkingMemory::new();

        let ranks =
            personalised_page_rank_optimised(&pr_graph, 0.85, &p_vec, 100, 1e-6, &mut memory);

        assert_eq!(ranks.len(), 3);
        assert!((ranks[1] - ranks[2]).abs() < 1e-6);
        assert!((ranks.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }
}
