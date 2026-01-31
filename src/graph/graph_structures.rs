use std::clone::Clone;

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
