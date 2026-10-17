//! Implementations of the PageRank algorithm for rapid identification of
//! influential nodes in the network

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::graph_structures::edges_to_csr;
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Default convergence threshold on the L1 change between iterations.
const DEFAULT_TOLERANCE: f64 = 1e-12;

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
#[derive(Clone)]
pub struct PageRankGraph<T> {
    /// The total number of nodes in the graph
    node_count: usize,
    /// Flattened incoming edges: `[node0_in_edges..., node1_in_edges..., ...]`
    in_edges_flat: Vec<usize>,
    /// Edge weight per entry of `in_edges_flat`
    in_edge_weights_flat: Vec<T>,
    /// Offsets into `in_edges_flat` for each node, length `node_count + 1`
    in_edges_offsets: Vec<usize>,
    /// Inverse of the summed out-weight per node, zero for dangling nodes
    inv_out_weight: Vec<T>,
}

impl<T> PageRankGraph<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    /// Generate the structure from 0-based edge indices.
    ///
    /// ### Params
    ///
    /// * `node_count` - Number of nodes
    /// * `from` - 0-based source node per edge
    /// * `to` - 0-based target node per edge
    /// * `weights` - Optional weight per edge. Defaults to 1.
    /// * `undirected` - Add every edge in both directions
    ///
    /// ### Returns
    ///
    /// Initialised `PageRankGraph` structure
    pub fn from_indices(
        node_count: usize,
        from: &[usize],
        to: &[usize],
        weights: Option<&[T]>,
        undirected: bool,
    ) -> Self {
        assert_same_len!(from, to);

        if let Some(weights) = weights {
            assert_same_len!(from, weights);
        }

        let mut out_edges: Vec<Vec<(usize, T)>> = vec![Vec::new(); node_count];
        let mut in_edges: Vec<Vec<(usize, T)>> = vec![Vec::new(); node_count];

        for (i, (&from_idx, &to_idx)) in from.iter().zip(to.iter()).enumerate() {
            let weight = weights.map(|w| w[i]).unwrap_or(T::one());

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

        let inv_out_weight = inv_out_weights(&out_edges);

        Self {
            node_count,
            in_edges_flat,
            in_edge_weights_flat,
            in_edges_offsets,
            inv_out_weight,
        }
    }

    /// Generate the structure directly from node names and edge lists
    ///
    /// ### Params
    ///
    /// * `nodes` - Slice of the node names
    /// * `from` - Slice of the names of the from nodes
    /// * `to` - Slice of the names of the to nodes
    /// * `weights` - Optional weight per edge. Defaults to 1.
    /// * `undirected` - Whether to create bidirectional edges
    ///
    /// ### Returns
    ///
    /// Initialised `PageRankGraph` structure
    ///
    /// ### Panics
    ///
    /// If an edge names a node that is not in `nodes`.
    pub fn from_strings(
        nodes: &[String],
        from: &[String],
        to: &[String],
        weights: Option<&[T]>,
        undirected: bool,
    ) -> Self {
        let name_to_idx: FxHashMap<&String, usize> = nodes
            .iter()
            .enumerate()
            .map(|(idx, name)| (name, idx))
            .collect();
        let lookup = |name: &String| {
            *name_to_idx
                .get(name)
                .unwrap_or_else(|| panic!("Node '{name}' not found in nodes list"))
        };
        let from_idx: Vec<usize> = from.iter().map(lookup).collect();
        let to_idx: Vec<usize> = to.iter().map(lookup).collect();

        Self::from_indices(nodes.len(), &from_idx, &to_idx, weights, undirected)
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

/// Inverse summed out-weight per node.
///
/// Nodes whose out-weight sums to zero (no out-edges, or only zero-weight
/// ones) get zero, so their mass leaks. The callers renormalise at the end,
/// which for personalised PageRank is the same as teleporting the dangling
/// mass back to the personalisation vector: the solution is only rescaled.
///
/// ### Params
///
/// * `out_edges` - Out-edges per node as `(target, weight)`
///
/// ### Returns
///
/// The inverse out-weight per node.
fn inv_out_weights<T>(out_edges: &[Vec<(usize, T)>]) -> Vec<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    out_edges
        .iter()
        .map(|edges| {
            let total: T = edges.iter().map(|(_, weight)| *weight).sum();
            if total > T::zero() {
                T::one() / total
            } else {
                T::zero()
            }
        })
        .collect()
}

//////////////
// PageRank //
//////////////

/// Parallel personalised PageRank algorithm for a single run.
///
/// Allocates its own working memory. For many runs on one graph, call
/// [personalised_page_rank_optimised] with thread-local working memory instead.
///
/// ### Params
///
/// * `graph` - The `PageRankGraph` on which to run the personalised page-rank.
/// * `damping_factor` - The dampening factor parameter, i.e., the probability
///   of continuing the walk.
/// * `personalisation_vector` - The vector of probabilities for the reset,
///   making this the personalised page rank.
/// * `nb_iter` - Maximum number of iterations for the personalised page rank.
/// * `tol` - Optional convergence threshold on the L1 change between
///   iterations. Defaults to `1e-12`.
///
/// ### Returns
///
/// The (normalised) personalised page rank scores.
pub fn personalised_page_rank<T>(
    graph: &PageRankGraph<T>,
    damping_factor: T,
    personalisation_vector: &[T],
    nb_iter: usize,
    tol: Option<T>,
) -> Vec<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    if graph.node_count == 0 {
        return vec![];
    }

    assert!(
        T::zero() <= damping_factor && damping_factor <= T::one(),
        "Damping factor should be between 0 and 1."
    );
    assert_eq!(
        personalisation_vector.len(),
        graph.node_count,
        "Personalisation vector length must match node count."
    );

    let tolerance = tol.unwrap_or(T::from_f64(DEFAULT_TOLERANCE).unwrap());

    personalised_page_rank_optimised(
        graph,
        damping_factor,
        personalisation_vector,
        nb_iter,
        tolerance,
        &mut PageRankWorkingMemory::new(),
    )
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
/// * `tolerance` - Convergence threshold on the L1 change between iterations.
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
                    .map(|(&w, &edge_weight)| ranks[w] * edge_weight * graph.inv_out_weight[w])
                    .sum();

                *new_rank = teleport_prob + damping_factor * link_prob;
            });

        let l1: T = new_ranks[..node_count]
            .par_iter()
            .zip(&ranks[..node_count])
            .map(|(new, old)| (*new - *old).abs())
            .sum();

        // Swap vectors (no allocation)
        std::mem::swap(ranks, new_ranks);

        if l1 <= tolerance {
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
/// * `graph` - The `AttributedGraph` with node and edge types
/// * `damping_factor` - The dampening factor parameter, i.e., the probability
///   of resetting.
/// * `personalization_vector` - The vector of probabilities for the reset,
///   making this the personalised page rank.
/// * `nb_iter` - Maximum number of iterations for the personalised page rank.
/// * `tol` - Optional convergence threshold on the L1 change between
///   iterations. Defaults to `1e-12`.
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
    graph: &AttributedGraph<T>,
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
    let node_count = graph.node_types.len();
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

    for (((&source, &target), &edge_type), &weight) in graph
        .from
        .iter()
        .zip(&graph.to)
        .zip(&graph.edge_types)
        .zip(&graph.weights)
    {
        // sink nodes have no valid outgoing edges
        if sink_types.contains(graph.node_types[source]) {
            continue;
        }

        let is_sink_edge = constrained_types.contains(edge_type);
        out_edges[source].push((target, weight, is_sink_edge));
        in_edges[target].push((source, weight, is_sink_edge));
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

                // zero-weight sources leak; the final normalisation restores it
                for &(w, edge_weight, is_sink_edge) in &in_edges[v] {
                    if out_degrees[w] > T::zero() {
                        let flow =
                            damping_factor * flowable_ranks[w] * edge_weight / out_degrees[w];

                        if is_sink_edge {
                            absorbed += flow;
                        } else {
                            flowable += flow;
                        }
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
        let l1 = total_ranks
            .par_iter()
            .zip(flowable_ranks.par_iter().zip(&absorbed_ranks))
            .map(|(new_total, (old_flow, old_abs))| (*new_total - (*old_flow + *old_abs)).abs())
            .sum::<T>();

        flowable_ranks = final_flowable;
        absorbed_ranks = new_absorbed;

        if l1 <= tolerance {
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

////////////////////////////////////////////
// Constrained page rank on a fixed graph //
////////////////////////////////////////////

/// Precomputed heterogeneous graph for repeated constrained personalised
/// PageRank runs, i.e. the diffusion profiles of Ruiz et al.
#[derive(Clone, Debug)]
pub struct ConstrainedPageRankGraph<T> {
    /// Number of nodes
    node_count: usize,
    /// Number of distinct node types
    n_types: usize,
    /// Type id per node
    node_type: Vec<u16>,
    /// Whether the node belongs to a sink type
    is_sink_type: Vec<bool>,
    /// Weight per type id. `None` uses the raw edge weights.
    type_weights: Option<Vec<T>>,
    /// In-edge offsets, length `node_count + 1`
    in_indptr: Vec<usize>,
    /// Source node per in-edge
    in_indices: Vec<u32>,
    /// Edge weight per in-edge
    in_weights: Vec<T>,
    /// Out-edge offsets, length `node_count + 1`
    out_indptr: Vec<usize>,
    /// Target node per out-edge
    out_indices: Vec<u32>,
    /// Edge weight per out-edge
    out_weights: Vec<T>,
    /// Transition factors without seeds, `node_count x n_types`, row-major
    base_factors: Vec<T>,
}

impl<T> ConstrainedPageRankGraph<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    /// Build the graph from 0-based edge indices and node types.
    ///
    /// ### Params
    ///
    /// * `node_types` - Type per node. Type ids follow order of first
    ///   appearance.
    /// * `from` - 0-based source node per edge
    /// * `to` - 0-based target node per edge
    /// * `weights` - Optional non-negative weight per edge. Defaults to 1.
    /// * `type_weights` - Optional weight per node type. Must cover every type
    ///   in `node_types`. `None` gives the plain random walk.
    /// * `sink_types` - Node types that act as sinks (no out-flow) unless the
    ///   node is a seed of the run. Unknown names are ignored.
    /// * `undirected` - Add every edge in both directions.
    ///
    /// ### Returns
    ///
    /// The initialised `ConstrainedPageRankGraph`, or an error for malformed
    /// input.
    pub fn new(
        node_types: &[String],
        from: &[usize],
        to: &[usize],
        weights: Option<&[T]>,
        type_weights: Option<&FxHashMap<String, T>>,
        sink_types: &FxHashSet<String>,
        undirected: bool,
    ) -> Result<Self, MethodsErrors> {
        let node_count = node_types.len();
        if node_count == 0 {
            return Err(CoreErrors::InvalidArgument("The graph has no nodes.".to_string()).into());
        }
        if node_count > u32::MAX as usize {
            return Err(CoreErrors::InvalidArgument(format!(
                "{node_count} nodes exceed the u32 index range."
            ))
            .into());
        }
        if to.len() != from.len() {
            return Err(MethodsErrors::GraphLengthMismatch {
                name: "to",
                expected: from.len(),
                got: to.len(),
            });
        }
        if let Some(w) = weights
            && w.len() != from.len()
        {
            return Err(MethodsErrors::GraphLengthMismatch {
                name: "weights",
                expected: from.len(),
                got: w.len(),
            });
        }

        // type ids in order of first appearance
        let mut type_ids: FxHashMap<&str, u16> = FxHashMap::default();
        let mut type_names: Vec<&str> = Vec::new();
        let mut node_type = Vec::with_capacity(node_count);
        for t in node_types {
            let id = match type_ids.get(t.as_str()) {
                Some(&id) => id,
                None => {
                    if type_names.len() >= u16::MAX as usize {
                        return Err(CoreErrors::InvalidArgument(
                            "Too many distinct node types.".to_string(),
                        )
                        .into());
                    }
                    let id = type_names.len() as u16;
                    type_ids.insert(t.as_str(), id);
                    type_names.push(t.as_str());
                    id
                }
            };
            node_type.push(id);
        }
        let n_types = type_names.len();

        let type_weights = match type_weights {
            Some(map) => {
                let mut w = Vec::with_capacity(n_types);
                for name in &type_names {
                    let w_t = *map
                        .get(*name)
                        .ok_or_else(|| MethodsErrors::MissingNodeTypeWeight(name.to_string()))?;
                    if !(w_t.is_finite() && w_t > T::zero()) {
                        return Err(CoreErrors::InvalidArgument(format!(
                            "The weight of node type '{name}' must be positive and finite."
                        ))
                        .into());
                    }
                    w.push(w_t);
                }
                Some(w)
            }
            None => None,
        };

        let is_sink_type: Vec<bool> = node_types.iter().map(|t| sink_types.contains(t)).collect();

        let n_edges = if undirected {
            2 * from.len()
        } else {
            from.len()
        };
        let mut src: Vec<u32> = Vec::with_capacity(n_edges);
        let mut dst: Vec<u32> = Vec::with_capacity(n_edges);
        let mut edge_w: Vec<T> = Vec::with_capacity(n_edges);
        for (k, (&a, &b)) in from.iter().zip(to).enumerate() {
            for (name, idx) in [("from", a), ("to", b)] {
                if idx >= node_count {
                    return Err(MethodsErrors::GraphIndexOutOfRange {
                        name,
                        index: idx,
                        n_nodes: node_count,
                    });
                }
            }
            let w_k = weights.map_or(T::one(), |w| w[k]);
            if !(w_k.is_finite() && w_k >= T::zero()) {
                return Err(CoreErrors::InvalidArgument(format!(
                    "Edge weight {k} must be non-negative and finite."
                ))
                .into());
            }
            src.push(a as u32);
            dst.push(b as u32);
            edge_w.push(w_k);
            if undirected {
                src.push(b as u32);
                dst.push(a as u32);
                edge_w.push(w_k);
            }
        }

        let (out_indptr, out_indices, out_weights) = edges_to_csr(node_count, &src, &dst, &edge_w);
        let (in_indptr, in_indices, in_weights) = edges_to_csr(node_count, &dst, &src, &edge_w);

        let mut graph = Self {
            node_count,
            n_types,
            node_type,
            is_sink_type,
            type_weights,
            in_indptr,
            in_indices,
            in_weights,
            out_indptr,
            out_indices,
            out_weights,
            base_factors: Vec::new(),
        };

        let no_seeds = vec![false; node_count];
        let mut base_factors = vec![T::zero(); node_count * n_types];
        base_factors
            .par_chunks_mut(n_types)
            .enumerate()
            .for_each(|(i, row)| graph.fill_factor_row(i, &no_seeds, row));
        graph.base_factors = base_factors;

        Ok(graph)
    }

    /// Number of nodes in the graph.
    ///
    /// ### Returns
    ///
    /// The node count.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Compute the transition factors of one node given the current seeds.
    ///
    /// Non-seed nodes of a sink type get an all-zero row. Edges into seeds are
    /// ignored, as the seeds are sources in the constrained graph.
    ///
    /// ### Params
    ///
    /// * `i` - Node index
    /// * `seed_mask` - Whether each node is a seed of the current run
    /// * `row` - Output slice of length `n_types`
    fn fill_factor_row(&self, i: usize, seed_mask: &[bool], row: &mut [T]) {
        row.fill(T::zero());
        if self.is_sink_type[i] && !seed_mask[i] {
            return;
        }

        for k in self.out_indptr[i]..self.out_indptr[i + 1] {
            let j = self.out_indices[k] as usize;
            if !seed_mask[j] {
                row[self.node_type[j] as usize] += self.out_weights[k];
            }
        }

        match &self.type_weights {
            Some(w) => {
                let denom: T = row
                    .iter()
                    .zip(w)
                    .filter(|(s, _)| **s > T::zero())
                    .map(|(_, w_t)| *w_t)
                    .sum();
                for (s, w_t) in row.iter_mut().zip(w) {
                    *s = if *s > T::zero() {
                        *w_t / (denom * *s)
                    } else {
                        T::zero()
                    };
                }
            }
            None => {
                let total: T = row.iter().copied().sum();
                let f = if total > T::zero() {
                    T::one() / total
                } else {
                    T::zero()
                };
                row.fill(f);
            }
        }
    }
}

/// Working memory for [constrained_personalised_page_rank_optimised]
///
/// One per thread; reused across runs to avoid per-seed allocations.
#[derive(Debug)]
pub struct ConstrainedPageRankWorkingMemory<T> {
    /// Current ranks
    ranks: Vec<T>,
    /// Ranks of the next iteration
    new_ranks: Vec<T>,
    /// Transition factors for the current seeds, `node_count x n_types`
    factors: Vec<T>,
    /// Whether each node is a seed of the current run
    seed_mask: Vec<bool>,
    /// Nodes whose factor rows differ from the seed-free base
    touched: Vec<usize>,
}

impl<T> ConstrainedPageRankWorkingMemory<T>
where
    T: BixverseFloat,
{
    /// Initialise empty working memory
    ///
    /// ### Returns
    ///
    /// Initialised `ConstrainedPageRankWorkingMemory`
    pub fn new() -> Self {
        Self {
            ranks: Vec::new(),
            new_ranks: Vec::new(),
            factors: Vec::new(),
            seed_mask: Vec::new(),
            touched: Vec::new(),
        }
    }
}

impl<T> Default for ConstrainedPageRankWorkingMemory<T>
where
    T: BixverseFloat,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Constrained personalised PageRank on a precomputed graph
///
/// Implements the diffusion profiles of Ruiz et al.: nodes with non-zero
/// personalisation are the seeds and act as sources (in-edges dropped), while
/// all other nodes of a sink type absorb mass (out-edges dropped). The rows of
/// the seeds and their in-neighbours are renormalised for this run; the rest
/// comes from the precomputed base factors.
///
/// The paper sends sink mass back to the seeds via `alpha * s * sum_J r_j`.
/// That term is a scalar multiple of `s`, so letting the mass leak and
/// normalising at the end gives the same profile. Same holds for dangling
/// nodes.
///
/// Sequential over nodes: the caller is expected to parallelise over seeds.
///
/// ### Params
///
/// * `graph` - The precomputed `ConstrainedPageRankGraph`
/// * `damping_factor` - Probability of continuing the walk (`alpha`)
/// * `personalisation_vector` - Non-negative restart distribution over the
///   nodes. Non-zero entries define the seeds.
/// * `max_iter` - Maximum number of power iterations
/// * `tolerance` - Convergence threshold on the L1 change between iterations
/// * `working_memory` - Reusable `ConstrainedPageRankWorkingMemory`
///
/// ### Returns
///
/// The normalised diffusion profile, or an error for invalid input.
///
/// ### References
///
/// Ruiz et al., Nat Commun, 2021
pub fn constrained_personalised_page_rank_optimised<T>(
    graph: &ConstrainedPageRankGraph<T>,
    damping_factor: T,
    personalisation_vector: &[T],
    max_iter: usize,
    tolerance: T,
    working_memory: &mut ConstrainedPageRankWorkingMemory<T>,
) -> Result<Vec<T>, MethodsErrors>
where
    T: BixverseFloat + std::iter::Sum,
{
    let n = graph.node_count;
    let nt = graph.n_types;

    if personalisation_vector.len() != n {
        return Err(MethodsErrors::GraphLengthMismatch {
            name: "personalisation_vector",
            expected: n,
            got: personalisation_vector.len(),
        });
    }
    if !(T::zero() <= damping_factor && damping_factor <= T::one()) {
        return Err(CoreErrors::InvalidArgument(
            "The damping factor must be within [0, 1].".to_string(),
        )
        .into());
    }
    if personalisation_vector
        .iter()
        .any(|p| !(p.is_finite() && *p >= T::zero()))
    {
        return Err(CoreErrors::InvalidArgument(
            "The personalisation vector must be non-negative and finite.".to_string(),
        )
        .into());
    }
    let p_sum: T = personalisation_vector.iter().copied().sum();
    if p_sum <= T::zero() {
        return Err(CoreErrors::InvalidArgument(
            "The personalisation vector sums to zero.".to_string(),
        )
        .into());
    }

    let ConstrainedPageRankWorkingMemory {
        ranks,
        new_ranks,
        factors,
        seed_mask,
        touched,
    } = working_memory;

    ranks.resize(n, T::zero());
    new_ranks.resize(n, T::zero());
    factors.clear();
    factors.extend_from_slice(&graph.base_factors);
    seed_mask.clear();
    seed_mask.extend(personalisation_vector.iter().map(|p| *p > T::zero()));

    // seeds and every node with an edge into a seed get new factor rows
    touched.clear();
    for s in (0..n).filter(|&s| seed_mask[s]) {
        touched.push(s);
        let (start, end) = (graph.in_indptr[s], graph.in_indptr[s + 1]);
        touched.extend(graph.in_indices[start..end].iter().map(|&w| w as usize));
    }
    for &i in touched.iter() {
        graph.fill_factor_row(i, seed_mask, &mut factors[i * nt..(i + 1) * nt]);
    }

    ranks.copy_from_slice(personalisation_vector);
    let teleport_factor = T::one() - damping_factor;

    for _ in 0..max_iter {
        for v in 0..n {
            let mut rank = teleport_factor * personalisation_vector[v];
            if !seed_mask[v] {
                let t_v = graph.node_type[v] as usize;
                for k in graph.in_indptr[v]..graph.in_indptr[v + 1] {
                    let w = graph.in_indices[k] as usize;
                    rank += damping_factor * ranks[w] * factors[w * nt + t_v] * graph.in_weights[k];
                }
            }
            new_ranks[v] = rank;
        }

        let l1: T = new_ranks
            .iter()
            .zip(ranks.iter())
            .map(|(a, b)| (*a - *b).abs())
            .sum();

        std::mem::swap(ranks, new_ranks);

        if l1 <= tolerance {
            break;
        }
    }

    let sum: T = ranks.iter().copied().sum();
    let mut res = ranks.clone();
    if sum > T::zero() {
        res.iter_mut().for_each(|x| *x /= sum);
    }

    Ok(res)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::graph_structures::graph_from_strings_with_attributes;

    /// Both PPR paths match igraph's prpack on a weighted directed graph with
    /// a dangling node (7) and a node with only a zero-weight out-edge (6).
    /// Reference from `igraph::page_rank(algo = "prpack", damping = 0.8)`,
    /// igraph 2.3.3.
    #[test]
    fn test_page_rank_matches_igraph_prpack() {
        let from: Vec<usize> = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6];
        let to: Vec<usize> = vec![1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 0];
        let weights = vec![0.5, 1.5, 2.0, 0.3, 1.0, 0.7, 0.9, 1.2, 2.5, 0.4, 0.8, 0.0];
        let expected = [
            2.845_972_704_682_356_3e-1,
            5.691_945_409_364_712_6e-2,
            2.103_545_042_591_307_1e-1,
            1.049_297_762_422_016_8e-1,
            1.052_691_717_448_718_7e-1,
            1.321_832_351_066_182_5e-1,
            3.524_886_269_509_82e-2,
            7.049_772_539_019_64e-2,
        ];
        let mut p_vec = vec![0.0; 8];
        p_vec[0] = 1.0;

        let graph = PageRankGraph::from_indices(8, &from, &to, Some(&weights), false);

        let tol = DEFAULT_TOLERANCE;
        let optimised = personalised_page_rank_optimised(
            &graph,
            0.8,
            &p_vec,
            1000,
            tol,
            &mut PageRankWorkingMemory::new(),
        );
        let reference = personalised_page_rank(&graph, 0.8, &p_vec, 1000, None);

        for i in 0..8 {
            assert!(
                (optimised[i] - expected[i]).abs() < 1e-10,
                "node {i}: optimised {} vs igraph {}",
                optimised[i],
                expected[i]
            );
            assert!(
                (reference[i] - expected[i]).abs() < 1e-10,
                "node {i}: reference {} vs igraph {}",
                reference[i],
                expected[i]
            );
        }
    }

    /// On the chain a -> b -> c with a sink edge a -> b, mass reaching b is
    /// absorbed there and never forwarded, so c only gets what b's other
    /// in-flows carry: nothing. Without the sink edge c gets mass.
    #[test]
    fn test_constrained_page_rank_sink_edge_absorbs_mass() {
        let nodes: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let node_types: Vec<String> = vec!["n".to_string(); 3];
        let from = vec!["a".to_string(), "b".to_string()];
        let to = vec!["b".to_string(), "c".to_string()];
        let edge_types = vec!["sink".to_string(), "flow".to_string()];
        let weights = vec![1.0, 1.0];
        let graph = graph_from_strings_with_attributes(
            &nodes,
            &node_types,
            &from,
            &to,
            &edge_types,
            &weights,
        );
        let p_vec = [1.0, 0.0, 0.0];
        let sink_edges: FxHashSet<String> = ["sink".to_string()].into_iter().collect();

        let constrained = constrained_personalised_page_rank(
            &graph,
            0.85,
            &p_vec,
            1000,
            None,
            None,
            Some(&sink_edges),
        );
        let free = constrained_personalised_page_rank(&graph, 0.85, &p_vec, 1000, None, None, None);

        assert!((constrained.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(constrained[1] > 0.0);
        assert_eq!(constrained[2], 0.0);
        assert!(free[2] > 0.0);
    }

    /// Toy multiscale graph: two drugs, three proteins, one function, one
    /// disease. Undirected edges.
    fn toy_multiscale() -> (Vec<String>, Vec<usize>, Vec<usize>, FxHashMap<String, f64>) {
        let types: Vec<String> = [
            "drug", "drug", "protein", "protein", "protein", "function", "disease",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let from = vec![0, 0, 1, 2, 3, 2, 4, 6];
        let to = vec![2, 3, 3, 3, 4, 5, 5, 4];
        let type_weights: FxHashMap<String, f64> = [
            ("drug", 3.0),
            ("protein", 2.0),
            ("function", 1.5),
            ("disease", 4.0),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
        (types, from, to, type_weights)
    }

    fn toy_sinks() -> FxHashSet<String> {
        ["drug", "disease"].iter().map(|s| s.to_string()).collect()
    }

    /// The precomputed constrained PageRank matches a dense power iteration of
    /// eqs 1-3 in Ruiz et al., including the explicit sink teleport.
    #[test]
    fn test_constrained_optimised_matches_dense_reference() {
        let (types, from, to, tw) = toy_multiscale();
        let sinks = toy_sinks();
        let n = types.len();
        let seed = 0;
        let alpha = 0.85;

        // dense G': undirected -> both directions, no in-edges to the seed, no
        // out-edges from other sinks
        let is_sink = |i: usize| sinks.contains(&types[i]) && i != seed;
        let mut adj = vec![vec![false; n]; n];
        for (&a, &b) in from.iter().zip(&to) {
            adj[a][b] = true;
            adj[b][a] = true;
        }
        for (i, row) in adj.iter_mut().enumerate() {
            row[seed] = false;
            if is_sink(i) {
                row.fill(false);
            }
        }
        let mut m = vec![vec![0.0; n]; n];
        for i in 0..n {
            let mut counts: FxHashMap<&str, f64> = FxHashMap::default();
            for j in (0..n).filter(|&j| adj[i][j]) {
                *counts.entry(types[j].as_str()).or_default() += 1.0;
            }
            let denom: f64 = counts.keys().map(|t| tw[*t]).sum();
            for j in (0..n).filter(|&j| adj[i][j]) {
                let t = types[j].as_str();
                m[i][j] = tw[t] / denom / counts[t];
            }
        }
        let mut s = vec![0.0; n];
        s[seed] = 1.0;
        let mut r = s.clone();
        for _ in 0..5000 {
            let sink_mass: f64 = (0..n).filter(|&j| is_sink(j)).map(|j| r[j]).sum();
            r = (0..n)
                .map(|v| {
                    let walk: f64 = (0..n).map(|i| r[i] * m[i][v]).sum();
                    (1.0 - alpha) * s[v] + alpha * s[v] * sink_mass + alpha * walk
                })
                .collect();
        }
        let total: f64 = r.iter().sum();
        r.iter_mut().for_each(|x| *x /= total);

        let graph =
            ConstrainedPageRankGraph::new(&types, &from, &to, None, Some(&tw), &sinks, true)
                .unwrap();
        let mut mem = ConstrainedPageRankWorkingMemory::new();
        let res = constrained_personalised_page_rank_optimised(
            &graph, alpha, &s, 10_000, 1e-14, &mut mem,
        )
        .unwrap();

        for (i, (a, b)) in r.iter().zip(&res).enumerate() {
            assert!((a - b).abs() < 1e-9, "node {i}: dense {a} vs optimised {b}");
        }
        assert!((res.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    /// The seed of sink type still diffuses, and other sinks receive mass.
    #[test]
    fn test_constrained_seed_is_source_not_sink() {
        let (types, from, to, tw) = toy_multiscale();
        let graph =
            ConstrainedPageRankGraph::new(&types, &from, &to, None, Some(&tw), &toy_sinks(), true)
                .unwrap();
        let mut mem = ConstrainedPageRankWorkingMemory::new();
        let mut s = vec![0.0; types.len()];
        s[0] = 1.0;
        let res =
            constrained_personalised_page_rank_optimised(&graph, 0.85, &s, 1000, 1e-12, &mut mem)
                .unwrap();

        assert!(res[0] < 0.5);
        assert!(res[1] > 0.0);
        assert!(res[6] > 0.0);
    }

    /// With type weights, every non-sink row of the seed-free transition matrix
    /// sums to one.
    #[test]
    fn test_constrained_type_weighted_rows_sum_to_one() {
        let (types, from, to, tw) = toy_multiscale();
        let graph =
            ConstrainedPageRankGraph::new(&types, &from, &to, None, Some(&tw), &toy_sinks(), true)
                .unwrap();
        let nt = graph.n_types;
        for i in 0..graph.node_count {
            let row_sum: f64 = (graph.out_indptr[i]..graph.out_indptr[i + 1])
                .map(|k| {
                    let j = graph.out_indices[k] as usize;
                    graph.base_factors[i * nt + graph.node_type[j] as usize] * graph.out_weights[k]
                })
                .sum();
            let expected = if graph.is_sink_type[i] { 0.0 } else { 1.0 };
            assert!((row_sum - expected).abs() < 1e-12, "node {i}: {row_sum}");
        }
    }

    /// Reusing working memory across seeds gives the same result as fresh
    /// memory.
    #[test]
    fn test_constrained_working_memory_reuse() {
        let (types, from, to, tw) = toy_multiscale();
        let graph =
            ConstrainedPageRankGraph::new(&types, &from, &to, None, Some(&tw), &toy_sinks(), true)
                .unwrap();
        let n = types.len();
        let mut s0 = vec![0.0; n];
        s0[0] = 1.0;
        let mut s6 = vec![0.0; n];
        s6[6] = 1.0;

        let mut mem = ConstrainedPageRankWorkingMemory::new();
        let _ =
            constrained_personalised_page_rank_optimised(&graph, 0.85, &s0, 1000, 1e-12, &mut mem)
                .unwrap();
        let reused =
            constrained_personalised_page_rank_optimised(&graph, 0.85, &s6, 1000, 1e-12, &mut mem)
                .unwrap();
        let fresh = constrained_personalised_page_rank_optimised(
            &graph,
            0.85,
            &s6,
            1000,
            1e-12,
            &mut ConstrainedPageRankWorkingMemory::new(),
        )
        .unwrap();

        assert_eq!(reused, fresh);
    }

    /// Malformed input is rejected with an error, not a panic.
    #[test]
    fn test_constrained_rejects_bad_input() {
        let (types, _, _, tw) = toy_multiscale();
        let sinks = toy_sinks();
        assert!(
            ConstrainedPageRankGraph::new(&types, &[0], &[99], None, Some(&tw), &sinks, true)
                .is_err()
        );
        let mut partial = tw.clone();
        partial.remove("function");
        assert!(
            ConstrainedPageRankGraph::new(&types, &[0], &[2], None, Some(&partial), &sinks, true)
                .is_err()
        );

        let graph =
            ConstrainedPageRankGraph::new(&types, &[0], &[2], None, Some(&tw), &sinks, true)
                .unwrap();
        let mut mem = ConstrainedPageRankWorkingMemory::new();
        let zeros = vec![0.0; types.len()];
        assert!(
            constrained_personalised_page_rank_optimised(&graph, 0.85, &zeros, 10, 1e-6, &mut mem)
                .is_err()
        );
    }
}
