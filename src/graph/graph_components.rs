//! Connected components over CSR adjacency matrices.
//!
//! Components are **weak**: a stored edge connects its endpoints regardless of
//! direction, matching `scipy.sparse.csgraph.connected_components` with
//! `directed=False`. Union-find is what makes that cheap on a directed CSR — it
//! unions each stored edge, so no transpose is needed to see the in-edges.
//!
//! Labels are assigned by scanning nodes in ascending order and numbering roots
//! on first encounter, which reproduces scipy's labelling.

use rustc_hash::FxHashMap;

use crate::graph::graph_structures::UnionFind;
use crate::prelude::*;

//////////
// Main //
//////////

/// Weakly connected components of a CSR adjacency.
///
/// ### Params
///
/// * `graph` - Square CSR adjacency. Edge direction is ignored.
///
/// ### Returns
///
/// `(n_components, labels)` where `labels[i]` is the zero-based component of
/// node `i`. Labels are assigned in ascending node order.
pub fn connected_components<T>(
    graph: &CompressedSparseData2<T>,
) -> Result<(usize, Vec<usize>), BixverseErrors>
where
    T: Clone,
{
    let n = validate_square_csr(graph)?;

    let mut uf = UnionFind::new(n);
    for i in 0..n {
        for idx in graph.indptr[i] as usize..graph.indptr[i + 1] as usize {
            uf.union(i as u32, graph.indices[idx]);
        }
    }

    Ok(label_by_root(
        &mut uf,
        (0..n as u32).collect::<Vec<_>>().as_slice(),
    ))
}

/// Weakly connected components of the subgraph induced by `members`.
///
/// Only edges with **both** endpoints in `members` are considered. Duplicated
/// entries in `members` are tolerated and land in the same component.
///
/// ### Params
///
/// * `graph` - Square CSR adjacency. Edge direction is ignored.
/// * `members` - Node indices spanning the induced subgraph.
///
/// ### Returns
///
/// `(n_components, labels)` where `labels[p]` is the component of
/// `members[p]`. Labels are assigned in ascending `members` position.
pub fn induced_components<T>(
    graph: &CompressedSparseData2<T>,
    members: &[usize],
) -> Result<(usize, Vec<usize>), BixverseErrors>
where
    T: Clone,
{
    let n = validate_square_csr(graph)?;

    let mut position: FxHashMap<usize, u32> = FxHashMap::default();
    for (p, &node) in members.iter().enumerate() {
        if node >= n {
            return Err(BixverseErrors::SliceIndexOutOfBounds {
                index: node,
                len: n,
            });
        }
        position.entry(node).or_insert(p as u32);
    }

    let mut uf = UnionFind::new(members.len());

    for (p, &node) in members.iter().enumerate() {
        // A repeated node shares a component with its first occurrence.
        if let Some(&first) = position.get(&node)
            && first != p as u32
        {
            uf.union(first, p as u32);
        }
        for idx in graph.indptr[node] as usize..graph.indptr[node + 1] as usize {
            let neighbour = graph.indices[idx] as usize;
            if let Some(&q) = position.get(&neighbour) {
                uf.union(p as u32, q);
            }
        }
    }

    Ok(label_by_root(
        &mut uf,
        (0..members.len() as u32).collect::<Vec<_>>().as_slice(),
    ))
}

/////////////
// Helpers //
/////////////

/// Turn a resolved union-find forest into dense component labels.
///
/// Roots are numbered on first encounter while scanning `elements` in order,
/// which is what gives ascending-node label assignment.
///
/// ### Params
///
/// * `uf` - The forest, mutated by the path compression inside `find`.
/// * `elements` - Elements to label, in the order the labels should follow.
///
/// ### Returns
///
/// `(n_components, labels)` aligned with `elements`.
fn label_by_root(uf: &mut UnionFind, elements: &[u32]) -> (usize, Vec<usize>) {
    let mut root_to_label: FxHashMap<u32, usize> = FxHashMap::default();
    let mut labels = Vec::with_capacity(elements.len());

    for &e in elements {
        let root = uf.find(e);
        let next = root_to_label.len();
        let label = *root_to_label.entry(root).or_insert(next);
        labels.push(label);
    }

    (root_to_label.len(), labels)
}

/// Check that a graph is a square CSR matrix and return its node count.
///
/// ### Params
///
/// * `graph` - The adjacency to validate.
///
/// ### Returns
///
/// The node count, or an error when the matrix is not CSR or not square.
fn validate_square_csr<T>(graph: &CompressedSparseData2<T>) -> Result<usize, BixverseErrors>
where
    T: Clone,
{
    if !graph.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }
    let (rows, cols) = graph.shape;
    if rows != cols {
        return Err(BixverseErrors::ShapeMismatch {
            expected: (rows, rows),
            got: (rows, cols),
        });
    }
    Ok(rows)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a CSR over `n` nodes storing exactly the given `(from, to)` edges,
    /// all at weight one.
    fn directed_csr(n: usize, edges: &[(usize, usize)]) -> CompressedSparseData2<f64> {
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &(a, b) in edges {
            rows[a].push(b as u32);
        }

        let mut indptr: Vec<u32> = vec![0];
        let mut indices: Vec<u32> = Vec::new();
        let mut data: Vec<f64> = Vec::new();
        for row in rows.iter_mut() {
            row.sort_unstable();
            for &j in row.iter() {
                indices.push(j);
                data.push(1.0);
            }
            indptr.push(indices.len() as u32);
        }

        CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (n, n))
    }

    /// Disjoint cliques come back as separate components with dense labels.
    #[test]
    fn test_components_of_two_cliques() {
        // {0,1,2} and {3,4} with no edge between them.
        let edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 3)];
        let graph = directed_csr(5, &edges);

        let (n_comps, labels) = connected_components(&graph).unwrap();

        assert_eq!(n_comps, 2);
        assert_eq!(labels, vec![0, 0, 0, 1, 1]);
    }

    /// Edge direction is ignored: this is weak connectivity, not strong.
    #[test]
    fn test_components_are_weak_not_strong() {
        // 0 -> 1 -> 2 with no return path: strongly this is three components,
        // weakly it is one.
        let graph = directed_csr(3, &[(0, 1), (1, 2)]);

        let (n_comps, labels) = connected_components(&graph).unwrap();

        assert_eq!(n_comps, 1);
        assert_eq!(labels, vec![0, 0, 0]);
    }

    /// Nodes with no edges are components in their own right, not dropped.
    #[test]
    fn test_isolated_nodes_get_own_components() {
        let graph = directed_csr(4, &[(0, 1)]);

        let (n_comps, labels) = connected_components(&graph).unwrap();

        assert_eq!(n_comps, 3);
        assert_eq!(labels, vec![0, 0, 1, 2]);
    }

    /// The induced subgraph is walked on its own, so removing a cut vertex
    /// splits it.
    #[test]
    fn test_induced_components_split_a_connected_graph() {
        // A path 0-1-2-3-4. Dropping node 2 splits it into {0,1} and {3,4}.
        let edges = [
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 4),
            (4, 3),
        ];
        let graph = directed_csr(5, &edges);

        let (n_comps, labels) = induced_components(&graph, &[0, 1, 3, 4]).unwrap();

        assert_eq!(n_comps, 2);
        assert_eq!(labels, vec![0, 0, 1, 1]);
    }

    /// Paths through non-members do not connect two members.
    #[test]
    fn test_induced_components_ignore_edges_leaving_the_subset() {
        // 0 and 2 are only connected through 1, which is not a member.
        let edges = [(0, 1), (1, 0), (1, 2), (2, 1)];
        let graph = directed_csr(3, &edges);

        let (n_comps, labels) = induced_components(&graph, &[0, 2]).unwrap();

        assert_eq!(n_comps, 2);
        assert_eq!(labels, vec![0, 1]);
    }

    /// Regression: a repeated member used to inflate the component count and
    /// give its copies different labels.
    #[test]
    fn test_induced_components_put_duplicate_members_in_one_component() {
        // Node 0 appears three times and node 2 twice, with no edge between the
        // two. Every copy must land on its first occurrence's label, and the
        // component count must not be inflated by the repeats.
        let edges = [(0, 1), (1, 0), (1, 2), (2, 1)];
        let graph = directed_csr(3, &edges);

        let (n_comps, labels) = induced_components(&graph, &[0, 2, 0, 2, 0]).unwrap();

        assert_eq!(n_comps, 2);
        assert_eq!(labels, vec![0, 1, 0, 1, 0]);
    }

    /// Regression: a duplicated member on a path used to split one component
    /// into two.
    #[test]
    fn test_induced_components_merge_through_a_duplicated_member() {
        // A path 0-1-2-3. Members list node 1 twice; the second copy must not
        // split the induced component in two.
        let edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)];
        let graph = directed_csr(4, &edges);

        let (n_comps, labels) = induced_components(&graph, &[0, 1, 2, 1]).unwrap();

        assert_eq!(n_comps, 1);
        assert_eq!(labels, vec![0, 0, 0, 0]);
    }

    /// A member index past the node count errors rather than panicking.
    #[test]
    fn test_induced_components_rejects_out_of_range_member() {
        let graph = directed_csr(3, &[(0, 1)]);

        assert!(induced_components(&graph, &[0, 9]).is_err());
    }

    /// A CSC matrix is refused rather than walked as if it were CSR.
    #[test]
    fn test_components_reject_csc_input() {
        let graph = directed_csr(3, &[(0, 1)]).transform();

        assert!(connected_components(&graph).is_err());
    }
}
