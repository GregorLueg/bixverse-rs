//! Minimum and maximum spanning forests over CSR adjacencies.
//!
//! Kruskal with union-find. The adjacency is read as **undirected**: a stored
//! entry connects its endpoints whichever direction it sits in, so a directed
//! CSR needs no transpose. Where a pair is stored in both directions with
//! different weights, the weight that favours the objective wins (smaller for
//! the minimum forest, larger for the maximum one).
//!
//! Disconnected input yields a forest rather than an error, matching
//! `scipy.sparse.csgraph.minimum_spanning_tree`. Two deliberate divergences
//! from that routine: explicitly stored zero weights are kept as real edges
//! (scipy treats a zero as an absent edge), and ties resolve by `(weight, lo,
//! hi)` so the output does not depend on traversal order.

use rayon::prelude::*;

use crate::core::math::sparse::undirected_edges_to_csr;
use crate::graph::graph_structures::UnionFind;
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Shared Kruskal core behind both public entry points.
///
/// Runs two sorts over the candidate edges. The first collapses each `(lo,
/// hi)` pair down to its objective-favouring weight; the second is the Kruskal
/// ordering itself. Sorting twice beats a hash map here because the candidate
/// list is one contiguous allocation the whole way through.
///
/// Those sorts are the whole cost, so they are the only part that runs in
/// parallel. Collecting the candidates is linear and the union-find pass is
/// inherently sequential. Rayon leaves short slices on the sequential path, so
/// this stays cheap on the tiny graphs that graph abstraction feeds it.
///
/// ### Params
///
/// * `graph` - Square CSR adjacency, read as undirected.
/// * `maximise` - Prefer the heaviest edges rather than the lightest.
///
/// ### Returns
///
/// The forest as a symmetric CSR, or an error when `graph` is not a square
/// CSR.
fn spanning_forest<T>(
    graph: &CompressedSparseData2<T>,
    maximise: bool,
) -> Result<CompressedSparseData2<T>, BixverseErrors>
where
    T: BixverseFloat + BixverseNumeric,
{
    let n = validate_square_csr(graph)?;

    let mut edges = candidate_edges(graph, n);

    // Objective-favouring weight first within each pair, so the dedup below
    // keeps the right one.
    edges.par_sort_unstable_by(|a, b| {
        (a.0, a.1)
            .cmp(&(b.0, b.1))
            .then_with(|| weight_order(a.2, b.2, maximise))
    });
    edges.dedup_by_key(|e| (e.0, e.1));

    edges.par_sort_unstable_by(|a, b| {
        weight_order(a.2, b.2, maximise).then_with(|| (a.0, a.1).cmp(&(b.0, b.1)))
    });

    let mut uf = UnionFind::new(n);
    let mut kept: Vec<(u32, u32, T)> = Vec::with_capacity(n.saturating_sub(1));
    for &(lo, hi, w) in &edges {
        if uf.union(lo, hi) {
            kept.push((lo, hi, w));
        }
    }

    kept.sort_unstable_by_key(|e| (e.0, e.1));

    Ok(undirected_edges_to_csr(n, &kept))
}

/// Collect the undirected candidate edges of a CSR adjacency.
///
/// Self loops and non-finite weights are dropped: neither can belong to a
/// spanning forest, and a `NaN` weight would poison the comparator.
///
/// ### Params
///
/// * `graph` - Square CSR adjacency.
/// * `n` - Node count.
///
/// ### Returns
///
/// `(lo, hi, weight)` triples with `lo < hi`, still holding both directions of
/// any pair stored twice.
fn candidate_edges<T>(graph: &CompressedSparseData2<T>, n: usize) -> Vec<(u32, u32, T)>
where
    T: BixverseFloat + BixverseNumeric,
{
    let mut edges: Vec<(u32, u32, T)> = Vec::with_capacity(graph.get_nnz());

    for i in 0..n {
        let row = i as u32;
        for idx in graph.indptr[i] as usize..graph.indptr[i + 1] as usize {
            let j = graph.indices[idx];
            let w = graph.data[idx];
            if j == row || !w.is_finite() {
                continue;
            }
            edges.push((row.min(j), row.max(j), w));
        }
    }

    edges
}

/// Order two weights so that the objective-favouring one compares as less.
///
/// ### Params
///
/// * `a` - First weight.
/// * `b` - Second weight.
/// * `maximise` - Heaviest-first rather than lightest-first.
///
/// ### Returns
///
/// The ordering to sort by. Uses `total_cmp`, so the comparator stays a total
/// order even though non-finite weights are already filtered out.
#[inline]
fn weight_order<T>(a: T, b: T, maximise: bool) -> std::cmp::Ordering
where
    T: BixverseFloat,
{
    if maximise {
        b.total_cmp(&a)
    } else {
        a.total_cmp(&b)
    }
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

//////////
// Main //
//////////

/// Minimum spanning forest of a CSR adjacency.
///
/// ### Params
///
/// * `graph` - Square CSR adjacency, read as undirected. Self loops and
///   non-finite weights are skipped.
///
/// ### Returns
///
/// A symmetric CSR of the same shape holding the retained edges at their
/// original weights, or an error when `graph` is not a square CSR.
pub fn minimum_spanning_forest<T>(
    graph: &CompressedSparseData2<T>,
) -> Result<CompressedSparseData2<T>, BixverseErrors>
where
    T: BixverseFloat + BixverseNumeric,
{
    spanning_forest(graph, false)
}

/// Maximum spanning forest of a CSR adjacency.
///
/// The same routine with the weight ordering flipped. Preferring the heaviest
/// edges is what PAGA's `connectivities_tree` wants, and it avoids the
/// reference implementation's detour through the reciprocals of the weights.
///
/// ### Params
///
/// * `graph` - Square CSR adjacency, read as undirected. Self loops and
///   non-finite weights are skipped.
///
/// ### Returns
///
/// A symmetric CSR of the same shape holding the retained edges at their
/// original weights, or an error when `graph` is not a square CSR.
pub fn maximum_spanning_forest<T>(
    graph: &CompressedSparseData2<T>,
) -> Result<CompressedSparseData2<T>, BixverseErrors>
where
    T: BixverseFloat + BixverseNumeric,
{
    spanning_forest(graph, true)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a symmetric CSR from an undirected weighted edge list.
    ///
    /// ### Params
    ///
    /// * `n` - Node count.
    /// * `edges` - Undirected edges as `(a, b, weight)`; both directions are
    ///   stored.
    ///
    /// ### Returns
    ///
    /// The CSR adjacency.
    fn undirected_csr(n: usize, edges: &[(usize, usize, f64)]) -> CompressedSparseData2<f64> {
        let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        for &(a, b, w) in edges {
            rows[a].push((b as u32, w));
            rows[b].push((a as u32, w));
        }

        let mut indptr: Vec<u32> = vec![0];
        let mut indices: Vec<u32> = Vec::new();
        let mut data: Vec<f64> = Vec::new();
        for row in rows.iter_mut() {
            row.sort_unstable_by_key(|e| e.0);
            for &(j, w) in row.iter() {
                indices.push(j);
                data.push(w);
            }
            indptr.push(indices.len() as u32);
        }

        CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (n, n))
    }

    /// Read a CSR back as a sorted `(lo, hi, weight)` edge list.
    ///
    /// ### Params
    ///
    /// * `csr` - The adjacency to read.
    ///
    /// ### Returns
    ///
    /// Deduplicated undirected edges, ascending by `(lo, hi)`.
    fn edge_list(csr: &CompressedSparseData2<f64>) -> Vec<(usize, usize, f64)> {
        let n = csr.nrows();
        let mut out = Vec::new();
        for i in 0..n {
            for idx in csr.indptr[i] as usize..csr.indptr[i + 1] as usize {
                let j = csr.indices[idx] as usize;
                if i < j {
                    out.push((i, j, csr.data[idx]));
                }
            }
        }
        out.sort_unstable_by_key(|e| (e.0, e.1));
        out
    }

    #[test]
    fn test_minimum_and_maximum_forests_pick_different_edges() {
        // A 4-cycle with one chord. Cheapest three edges are 0-1, 1-2, 2-3.
        // Going the other way, 0-2 and 0-3 are taken first, which leaves 2-3
        // closing a cycle, so node 1 gets attached through 1-2 instead.
        let graph = undirected_csr(
            4,
            &[
                (0, 1, 1.0),
                (1, 2, 2.0),
                (2, 3, 3.0),
                (0, 3, 9.0),
                (0, 2, 10.0),
            ],
        );

        let min_tree = edge_list(&minimum_spanning_forest(&graph).unwrap());
        assert_eq!(min_tree, vec![(0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0)]);

        let max_tree = edge_list(&maximum_spanning_forest(&graph).unwrap());
        assert_eq!(max_tree, vec![(0, 2, 10.0), (0, 3, 9.0), (1, 2, 2.0)]);
    }

    #[test]
    fn test_forest_spans_each_component_separately() {
        // Two triangles with no edge between them: 4 kept edges, not 5.
        let graph = undirected_csr(
            6,
            &[
                (0, 1, 1.0),
                (1, 2, 2.0),
                (0, 2, 3.0),
                (3, 4, 1.0),
                (4, 5, 2.0),
                (3, 5, 3.0),
            ],
        );

        let forest = edge_list(&minimum_spanning_forest(&graph).unwrap());

        assert_eq!(forest.len(), 4);
        assert_eq!(
            forest,
            vec![(0, 1, 1.0), (1, 2, 2.0), (3, 4, 1.0), (4, 5, 2.0)]
        );
    }

    #[test]
    fn test_isolated_node_contributes_no_edges() {
        let graph = undirected_csr(3, &[(0, 1, 1.0)]);

        let forest = minimum_spanning_forest(&graph).unwrap();

        assert_eq!(edge_list(&forest), vec![(0, 1, 1.0)]);
        assert_eq!(forest.shape, (3, 3));
        assert_eq!(forest.indptr[3], 2);
    }

    #[test]
    fn test_asymmetric_pair_resolves_towards_the_objective() {
        // 0->1 stored at 5.0, 1->0 at 1.0. The minimum forest must take 1.0
        // and the maximum forest 5.0.
        let data = vec![5.0f64, 1.0];
        let indices = vec![1u32, 0];
        let indptr = vec![0u32, 1, 2];
        let graph = CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (2, 2));

        assert_eq!(
            edge_list(&minimum_spanning_forest(&graph).unwrap()),
            vec![(0, 1, 1.0)]
        );
        assert_eq!(
            edge_list(&maximum_spanning_forest(&graph).unwrap()),
            vec![(0, 1, 5.0)]
        );
    }

    #[test]
    fn test_all_equal_weights_break_ties_deterministically() {
        // Every edge of K4 at the same weight: the tie-break is (lo, hi), so
        // the star at node 0 wins and the result is stable across runs.
        let mut edges = Vec::new();
        for i in 0..4 {
            for j in (i + 1)..4 {
                edges.push((i, j, 1.0));
            }
        }
        let graph = undirected_csr(4, &edges);

        let first = edge_list(&minimum_spanning_forest(&graph).unwrap());
        let second = edge_list(&minimum_spanning_forest(&graph).unwrap());

        assert_eq!(first, second);
        assert_eq!(first, vec![(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)]);
    }

    #[test]
    fn test_self_loops_and_non_finite_weights_are_dropped() {
        let data = vec![f64::NAN, 7.0, 1.0, 1.0];
        let indices = vec![0u32, 1, 0, 0];
        let indptr = vec![0u32, 2, 3, 4];
        // Row 0: self loop at NaN plus 0-1 at 7.0. Row 1: 1-0 at 1.0. Row 2:
        // 2-0 at 1.0. The NaN self loop must not survive, and 0-1 resolves to
        // the smaller stored weight.
        let graph = CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (3, 3));

        assert_eq!(
            edge_list(&minimum_spanning_forest(&graph).unwrap()),
            vec![(0, 1, 1.0), (0, 2, 1.0)]
        );
    }

    #[test]
    fn test_zero_weight_edges_are_kept() {
        // scipy's csgraph would treat the stored zero as an absent edge. Here
        // it is the cheapest edge in the graph and must win.
        let graph = undirected_csr(3, &[(0, 1, 0.0), (1, 2, 1.0), (0, 2, 2.0)]);

        assert_eq!(
            edge_list(&minimum_spanning_forest(&graph).unwrap()),
            vec![(0, 1, 0.0), (1, 2, 1.0)]
        );
    }

    #[test]
    fn test_empty_graph_returns_empty_forest() {
        let graph = CompressedSparseData2::<f64>::new_csr(&[], &[], &[0], None, (0, 0));

        let forest = minimum_spanning_forest(&graph).unwrap();

        assert_eq!(forest.shape, (0, 0));
        assert_eq!(forest.get_nnz(), 0);
    }

    #[test]
    fn test_rejects_csc_input() {
        let graph = undirected_csr(3, &[(0, 1, 1.0)]).transform();

        assert!(minimum_spanning_forest(&graph).is_err());
    }

    #[test]
    fn test_rejects_non_square_input() {
        let data = vec![1.0f64];
        let indices = vec![1u32];
        let indptr = vec![0u32, 1, 1];
        let graph = CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (2, 3));

        assert!(maximum_spanning_forest(&graph).is_err());
    }
}
