#![allow(dead_code)]

use faer::{Mat, MatRef};
use petgraph::Graph;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::clone::Clone;
use std::collections::BinaryHeap;

use crate::core::math::sparse::coo_to_csr;
use crate::prelude::*;

/////////////////////////////
// Sparse graph structures //
/////////////////////////////

/// Structure representation of a sparse graph
///
/// ### Fields
///
/// * `adjacency` - Sparse CSR representation of the graph with `f16` as
///   weights.
/// * `num_nodes` - Number of nodes represented in the graph.
/// * `directed` - Whether the graph is directed or not.
#[derive(Clone, Debug)]
pub struct SparseGraph<T: Clone> {
    /// Adjacency matrix in CSR (symmetric for undirected graphs)
    adjacency: CompressedSparseData<T>,
    num_nodes: usize,
    directed: bool,
}

impl<T> SparseGraph<T>
where
    T: Clone + BixverseFloat + std::iter::Sum,
{
    pub fn new(num_nodes: usize, adjacency: CompressedSparseData<T>, directed: bool) -> Self {
        Self {
            adjacency,
            num_nodes,
            directed,
        }
    }

    /// Helper function to get the neighbours and weights
    ///
    /// ### Params
    ///
    /// * `node` - Index of the node for which to get the neighbours
    ///
    /// ### Return
    ///
    /// Tuple of `(neighbour_indices, edge_weights)`
    #[inline]
    pub fn get_neighbours(&self, node: usize) -> (&[usize], &[T]) {
        let start = self.adjacency.indptr[node];
        let end = self.adjacency.indptr[node + 1];
        (
            &self.adjacency.indices[start..end],
            &self.adjacency.data[start..end],
        )
    }

    /// Get the node degree
    ///
    /// ### Params
    ///
    /// * `node` - Index of the node to get the node degree from
    ///
    /// ### Return
    ///
    /// The node degree for this node.
    #[inline]
    pub fn get_node_degree(&self, node: usize) -> usize {
        self.adjacency.indptr[node + 1] - self.adjacency.indptr[node]
    }

    /// Get total weight
    ///
    /// ### Params
    ///
    /// * `undirected` - Is the graph undirected or not.
    ///
    /// ### Returns
    ///
    /// The total edge weight
    pub fn total_weight(&self) -> T {
        let mut total = T::zero();
        for i in 0..self.num_nodes {
            let (_, weights) = self.get_neighbours(i);
            total += weights.iter().copied().sum();
        }
        let half = T::from_f64(0.5).unwrap();
        if !self.directed { total * half } else { total }
    }

    /// Expose the number of nodes
    ///
    /// ### Returns
    ///
    /// The number of nodes in the graph
    pub fn get_node_number(&self) -> usize {
        self.num_nodes
    }

    /// Expose if graph is directed
    ///
    /// ### Returns
    ///
    /// Boolean indicating if graph is directed
    pub fn is_directed(&self) -> bool {
        self.directed
    }
}

//////////////
// Petgraph //
//////////////

/// Generate a PetGraph Graph
///
/// ### Params
///
/// * `nodes` - Slice of the node names.
/// * `from` - Slice of the names of the from nodes.
/// * `to` - Slice of the names of the no nodes.
/// * `undirected` - Shall a directed or undirected graph be generated.
///
/// ### Returns
///
/// The generated PetGraph
pub fn graph_from_strings<'a, T>(
    nodes: &'a [String],
    from: &[String],
    to: &[String],
    weights: Option<&[T]>,
    undirected: bool,
) -> Graph<&'a str, T>
where
    T: BixverseFloat,
{
    assert_same_len!(from, to);

    if let Some(weights) = weights {
        assert_same_len!(from, weights);
    }

    let mut graph: Graph<&'a str, T> = Graph::new();

    let mut term_to_idx = FxHashMap::default();

    for term in nodes {
        let idx = graph.add_node(term);
        term_to_idx.insert(term, idx);
    }

    for (i, (from, to)) in from.iter().zip(to.iter()).enumerate() {
        let from = *term_to_idx.get(from).unwrap();
        let to = *term_to_idx.get(to).unwrap();
        let weight = weights.map(|w| w[i]).unwrap_or(T::one());
        graph.add_edge(from, to, weight);
        if undirected {
            graph.add_edge(to, from, weight);
        }
    }

    graph
}

//////////////////////////////////
// Hierachical graph structures //
//////////////////////////////////

/// Represents one level in the hierarchy
///
/// ### Fields
///
/// * `graph` - The sparse graph
/// * `node_map` - Maps fine node -> coarse node
/// * `coarse_weights` - Number of fine nodes that mapped to each coarse node
pub struct CoarseLevel<T>
where
    T: Clone,
{
    graph: SparseGraph<T>,
    node_map: Vec<usize>,
    coarse_weights: Vec<usize>,
}

/// Multi-level graph hierarchy
///
/// ### Fields
///
/// * `finest` - Original finest level graph
/// * `levels` - Coarse levels
pub struct GraphHierarchy<T>
where
    T: Clone,
{
    finest: SparseGraph<T>,
    levels: Vec<CoarseLevel<T>>,
}

impl<T> GraphHierarchy<T>
where
    T: Clone + BixverseFloat + std::iter::Sum + std::default::Default,
{
    /// Build a Graph hierarchy based on an initial graph
    ///
    /// This one assumes undirected graphs!!!
    ///
    /// ### Params
    ///
    /// * `graph` - The initial SparseGraph
    /// * `reduction_ratio` - By how much to reduce per level the number of
    ///   nodes. Typically 0.5.
    /// * `max_levels` - Maximum depth in terms of coarsion.
    ///
    /// ### Returns
    ///
    /// Initialised self.
    pub fn build(graph: SparseGraph<T>, reduction_ratio: T, max_levels: usize) -> Self {
        assert!(
            !graph.is_directed(),
            "This is implemented for undirected graphs!"
        );
        let mut levels = Vec::new();
        let mut current_graph = graph.clone();

        let min_nodes = T::from_usize(100).unwrap();

        for _ in 0..max_levels {
            let coarse_level = Self::coarsen_level(&current_graph);

            let current_nodes = T::from_usize(current_graph.num_nodes).unwrap();
            let coarse_nodes = T::from_usize(coarse_level.graph.num_nodes).unwrap();
            let reduction = coarse_nodes / current_nodes;

            if reduction > reduction_ratio || coarse_nodes < min_nodes {
                break;
            }

            current_graph = coarse_level.graph.clone();
            levels.push(coarse_level);
        }

        Self {
            finest: graph,
            levels,
        }
    }

    /// Coarsen one level using heavy-edge matching
    ///
    /// Generates a CoarseLevel of a given graph via heavy edge matching.
    ///
    /// ### Params
    ///
    /// * `graph` - The SparseGraph to coarsen
    ///
    /// ### Returns
    ///
    /// CoarseLevel containing the coarsened graph and mapping information
    fn coarsen_level(graph: &SparseGraph<T>) -> CoarseLevel<T> {
        let matching = Self::heavy_edge_matching(graph);
        let (coarse_graph, node_map, coarse_weights) = Self::contract_graph(graph, &matching);

        CoarseLevel {
            graph: coarse_graph,
            node_map,
            coarse_weights,
        }
    }

    /// Match nodes via heavy edges
    ///
    /// Finds pairs of nodes to match based on heaviest edge weights. Each node
    /// is matched with its heaviest unmatched neighbour.
    ///
    /// ### Params
    ///
    /// * `graph` - SparseGraph for which to identify the matching nodes
    ///
    /// ### Returns
    ///
    /// Vector where each element is either None (unmatched) or Some(node_id)
    /// indicating which node this node is matched with
    fn heavy_edge_matching(graph: &SparseGraph<T>) -> Vec<Option<usize>> {
        let n = graph.num_nodes;
        let mut matching = vec![None; n];
        let mut matched = vec![false; n];

        for i in 0..n {
            if matched[i] {
                continue;
            }

            let (neighbour, weights) = graph.get_neighbours(i);

            // Find heaviest unmatched neighbour
            let mut best_neighbour = None;
            let mut best_weight = T::zero();

            for (&j, &w) in neighbour.iter().zip(weights.iter()) {
                if !matched[j] && j != i && w > best_weight {
                    best_weight = w;
                    best_neighbour = Some(j)
                }
            }

            if let Some(j) = best_neighbour {
                matching[i] = Some(j);
                matching[j] = Some(i);
                matched[i] = true;
                matched[j] = true;
            }
        }

        matching
    }

    /// Contract graph based on matching
    ///
    /// Creates a coarser graph by merging matched nodes. Edges between matched
    /// nodes are removed, and edges to external nodes are combined.
    ///
    /// ### Params
    ///
    /// * `graph` - The fine-level graph to contract
    /// * `matching` - Node matching information from heavy_edge_matching
    ///
    /// ### Returns
    ///
    /// Tuple of (coarse_graph, node_map, coarse_weights) where node_map
    /// indicates which coarse node each fine node maps to, and coarse_weights
    /// counts how many fine nodes were merged into each coarse node
    fn contract_graph(
        graph: &SparseGraph<T>,
        matching: &[Option<usize>],
    ) -> (SparseGraph<T>, Vec<usize>, Vec<usize>) {
        let n = graph.num_nodes;
        let mut node_map = vec![0; n];
        let mut coarse_id = 0;

        // assign coarse node IDs
        for i in 0..n {
            if node_map[i] == 0 || i == 0 {
                match matching[i] {
                    Some(j) if j > i => {
                        node_map[i] = coarse_id;
                        node_map[j] = coarse_id;
                        coarse_id += 1;
                    }
                    None => {
                        // i unchanged
                        node_map[i] = coarse_id;
                        coarse_id += 1;
                    }
                    _ => {} // already assigned
                }
            }
        }

        let num_coarse = coarse_id;
        let mut coarse_weights = vec![0; num_coarse];

        for i in 0..n {
            coarse_weights[node_map[i]] += 1;
        }

        // Build coarse graph edges
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for i in 0..n {
            let ci = node_map[i];
            let (neighbours, weights) = graph.get_neighbours(i);

            for (&j, &w) in neighbours.iter().zip(weights.iter()) {
                let cj = node_map[j];
                if ci != cj {
                    rows.push(ci);
                    cols.push(cj);
                    vals.push(w);
                }
            }
        }

        // Sort to group duplicates
        let mut edges: Vec<(usize, usize, T)> = rows
            .into_iter()
            .zip(cols)
            .zip(vals)
            .map(|((r, c), v)| (r, c, v))
            .collect();
        edges.sort_unstable_by_key(|(r, c, _)| (*r, *c));

        // Sum duplicates
        let mut deduped_rows = Vec::new();
        let mut deduped_cols = Vec::new();
        let mut deduped_vals = Vec::new();

        if !edges.is_empty() {
            let (mut curr_r, mut curr_c, mut curr_v) = edges[0];

            for &(r, c, v) in &edges[1..] {
                if r == curr_r && c == curr_c {
                    curr_v += v;
                } else {
                    deduped_rows.push(curr_r);
                    deduped_cols.push(curr_c);
                    deduped_vals.push(curr_v);
                    (curr_r, curr_c, curr_v) = (r, c, v);
                }
            }
            deduped_rows.push(curr_r);
            deduped_cols.push(curr_c);
            deduped_vals.push(curr_v);
        }

        let coarse_csr = coo_to_csr(
            &deduped_rows,
            &deduped_cols,
            &deduped_vals,
            (num_coarse, num_coarse),
        );

        let coarse_graph = SparseGraph::new(num_coarse, coarse_csr, false);

        (coarse_graph, node_map, coarse_weights)
    }

    /// Project coarse solution to fine level
    ///
    /// ### Params
    ///
    /// * `level` - Which coarse level (0 = first coarsening)
    /// * `coarse_labels` - Labels at coarse level
    ///
    /// ### Returns
    ///
    /// Vector of labels at the fine level
    pub fn prolong(&self, level: usize, coarse_labels: &[usize]) -> Vec<usize> {
        if level >= self.levels.len() {
            return coarse_labels.to_vec();
        }

        let node_map = &self.levels[level].node_map;
        node_map.iter().map(|&ci| coarse_labels[ci]).collect()
    }

    /// Get coarsest graph
    ///
    /// ### Returns
    ///
    /// Reference to the coarsest graph in the hierarchy
    pub fn coarsest(&self) -> &SparseGraph<T> {
        self.levels.last().map(|l| &l.graph).unwrap_or(&self.finest)
    }
}

/////////////////////////
// Heterogenous graphs //
/////////////////////////

/// NodeData structure
///
/// ### Fields
///
/// * `name` - name of the node.
/// * `node_type` - type of the node
#[derive(Debug, Clone)]
#[allow(dead_code)] // clippy is wrongly complaining here
pub struct NodeData<'a> {
    pub name: &'a str,
    pub node_type: &'a str,
}

/// EdgeData structure
///
/// ### Fields
///
/// * `edge_type` - type of the edge.
/// * `weight` - weight of the edge.
#[derive(Debug, Clone)]
pub struct EdgeData<'a, T> {
    pub edge_type: &'a str,
    pub weight: &'a T,
}

/// Generate a weighted, labelled PetGraph Graph
///
/// ### Params
///
/// * `nodes` - Slice of node names.
/// * `node_types` - Slice of the node types.
/// * `from` - Slice of the names of the from nodes.
/// * `to` - Slice of the names of the no nodes.
/// * `edge_types` - Slice of the edge types.
/// * `edge_weights` - Slice of the edge weights.
///
/// ### Returns
///
/// A `Graph<NodeData<'_>, EdgeData<'_>>` graph for subsequent usage in
/// constraint personalised page-rank iteration.
///
/// ### Panics
///
/// Function will panic if `nodes` and `nodes_types` do not have the same
/// length and/or when `from`, `to`, `edge_types` and `edge_weights` do not
/// have the same length.
pub fn graph_from_strings_with_attributes<'a, T>(
    nodes: &'a [String],
    node_types: &'a [String],
    from: &'a [String],
    to: &'a [String],
    edge_types: &'a [String],
    edge_weights: &'a [T],
) -> Graph<NodeData<'a>, EdgeData<'a, T>>
where
    T: BixverseFloat,
{
    assert_same_len!(nodes, node_types);
    assert_same_len!(from, to, edge_types, edge_weights);

    let mut graph: Graph<NodeData<'_>, EdgeData<'_, T>> = Graph::new();
    let mut name_to_idx = FxHashMap::default();

    // add the nodes
    for (name, node_type) in nodes.iter().zip(node_types.iter()) {
        let node_data = NodeData { name, node_type };
        let idx = graph.add_node(node_data);
        name_to_idx.insert(name, idx);
    }

    // add the edges
    for (((from_name, to_name), edge_type), weight) in from
        .iter()
        .zip(to.iter())
        .zip(edge_types.iter())
        .zip(edge_weights.iter())
    {
        let from_idx = *name_to_idx
            .get(from_name)
            .unwrap_or_else(|| panic!("From node '{}' not found in nodes list", from_name));

        let to_idx = *name_to_idx
            .get(to_name)
            .unwrap_or_else(|| panic!("From node '{}' not found in nodes list", from_name));

        let edge_data = EdgeData { edge_type, weight };

        graph.add_edge(from_idx, to_idx, edge_data);
    }

    graph
}

///////////////////////////////
// Structure transformations //
///////////////////////////////

/// Generate a KNN graph adjacency matrix from a similarity matrix
///
/// ### Params
///
/// * `similarities` - The symmetric similarity matrix.
/// * `k` - Number of neighbours to take
///
/// ### Returns
///
/// The KNN adjacency matrix
pub fn get_knn_graph_adj<T>(similarities: &MatRef<T>, k: usize) -> Mat<T>
where
    T: BixverseFloat,
{
    assert_symmetric_mat!(similarities);
    let n = similarities.nrows();
    let mut adjacency: Mat<T> = Mat::zeros(n, n);

    // Parallelize across rows
    let rows: Vec<Vec<(usize, T)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut heap = BinaryHeap::with_capacity(k + 1);

            // Use min-heap to keep top-k similarities
            for j in 0..n {
                if i != j {
                    let sim = similarities[(i, j)];
                    heap.push((OrderedFloat(sim), j));
                    if heap.len() > k {
                        heap.pop(); // Remove smallest
                    }
                }
            }

            heap.into_iter()
                .map(|(ordered_sim, index)| (index, ordered_sim.0))
                .collect()
        })
        .collect();

    // Fill adjacency matrix
    for (i, neighbors) in rows.iter().enumerate() {
        for &(j, sim) in neighbors {
            adjacency[(i, j)] = sim;
        }
    }

    // Symmetrize in parallel
    let two = T::from_f64(2.0).unwrap();
    for i in 0..n {
        for j in i + 1..n {
            let val = (adjacency[(i, j)] + adjacency[(j, i)]) / two;
            adjacency[(i, j)] = val;
            adjacency[(j, i)] = val;
        }
    }

    adjacency
}

/// Generate a Laplacian matrix from an adjacency matrix
///
/// ### Params
///
/// * `adjacency` - The symmetric adjacency matrix.
///
/// ### Returns
///
/// The Laplacian matrix
pub fn adjacency_to_laplacian<T>(adjacency: &MatRef<T>, normalise: bool) -> Mat<T>
where
    T: BixverseFloat,
{
    assert_symmetric_mat!(adjacency);
    let n = adjacency.nrows();

    let degrees: Vec<T> = (0..n)
        .map(|i| {
            adjacency
                .row(i)
                .iter()
                .copied()
                .fold(T::zero(), |acc, x| acc + x)
        })
        .collect();

    if !normalise {
        let mut laplacian = adjacency.cloned();
        for i in 0..n {
            laplacian[(i, i)] = degrees[i] - adjacency[(i, i)];
            for j in 0..n {
                if i != j {
                    laplacian[(i, j)] = -adjacency[(i, j)];
                }
            }
        }
        laplacian
    } else {
        let inv_sqrt_d: Vec<T> = degrees
            .iter()
            .map(|&d| {
                if d > T::from_f64(1e-10).unwrap() {
                    T::one() / d.sqrt()
                } else {
                    T::zero()
                }
            })
            .collect();
        let mut laplacian = Mat::zeros(n, n);
        for i in 0..n {
            laplacian[(i, i)] = T::one();
            for j in 0..n {
                laplacian[(i, j)] -= inv_sqrt_d[i] * adjacency[(i, j)] * inv_sqrt_d[j];
            }
        }
        laplacian
    }
}

/// Convert kNN indices to undirected SparseGraph
///
/// ### Params
///
/// * `knn` - kNN indices. Excludes self.
///
/// ### Returns
///
/// Undirected sparse graph with symmetric CSR representation
pub fn knn_to_sparse_graph<T>(knn: &[Vec<usize>]) -> SparseGraph<T>
where
    T: BixverseFloat + std::iter::Sum + BixverseNumeric,
{
    let n_nodes = knn.len();
    let mut edges = Vec::new();

    // Collect all edges in both directions
    for (i, neighbours) in knn.iter().enumerate() {
        for &j in neighbours {
            edges.push((i, j));
            edges.push((j, i)); // Symmetric
        }
    }

    // Sort to group duplicates
    edges.sort_unstable();

    // Deduplicate and sum weights
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    if !edges.is_empty() {
        let (mut curr_r, mut curr_c) = edges[0];
        let mut weight = T::one();

        for &(r, c) in &edges[1..] {
            if r == curr_r && c == curr_c {
                weight += T::one();
            } else {
                rows.push(curr_r);
                cols.push(curr_c);
                vals.push(weight);
                (curr_r, curr_c) = (r, c);
                weight = T::one();
            }
        }
        rows.push(curr_r);
        cols.push(curr_c);
        vals.push(weight);
    }

    let adjacency = coo_to_csr(&rows, &cols, &vals, (n_nodes, n_nodes));

    SparseGraph::new(n_nodes, adjacency, false)
}
