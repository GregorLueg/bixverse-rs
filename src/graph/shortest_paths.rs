//! Shortest-path algorithms over CSR adjacency matrices.
//!
//! Dijkstra with lazy deletion, plus a rayon fan-out over many sources. The
//! routines take a [CompressedSparseData2] directly rather than a
//! [crate::prelude::SparseGraph], because the latter's neighbour accessor
//! allocates a `Vec` per call, which is fatal in a shortest-path inner loop.
//!
//! Distances accumulate in `f64` regardless of the edge weight type: a geodesic
//! is a sum over potentially thousands of edges, and `f32` accumulation loses
//! roughly four significant digits over that many terms.

use rayon::prelude::*;
use std::collections::BinaryHeap;

use crate::prelude::*;

//////////////
// Dijkstra //
//////////////

/// Single-source Dijkstra over a CSR adjacency with non-negative weights.
///
/// Uses a binary heap with lazy deletion: stale entries are pushed rather than
/// decreased in place, and skipped on pop. That trades a larger heap for
/// avoiding an index-into-heap structure, which is the faster arrangement for
/// the sparse graphs this crate deals with.
///
/// Nodes unreachable from `source` keep `f64::INFINITY`.
///
/// Edges whose weight is not finite (`NaN` or infinite) are **skipped**. `NaN`
/// in particular has to be dropped explicitly: it fails every comparison, so a
/// `NaN` candidate never relaxes and the endpoint would silently come back as
/// unreachable rather than as an error the caller can see.
///
/// ### Params
///
/// * `graph` - Square CSR adjacency. Edge weights must be non-negative;
///   negative weights silently produce wrong answers, as with any Dijkstra.
/// * `source` - Index of the source node.
/// * `dist` - Output buffer of length `graph.nrows()`. Overwritten in full, so
///   it may be reused across calls without clearing.
///
/// ### Returns
///
/// `Ok(())`, or an error when `graph` is not CSR, is not square, or `source`
/// is out of range.
pub fn dijkstra_from_source<T>(
    graph: &CompressedSparseData2<T>,
    source: usize,
    dist: &mut [f64],
) -> Result<(), BixverseErrors>
where
    T: BixverseFloat,
{
    let n = validate_graph(graph)?;

    if source >= n {
        return Err(BixverseErrors::SliceIndexOutOfBounds {
            index: source,
            len: n,
        });
    }
    if dist.len() != n {
        return Err(BixverseErrors::DimensionMisMatchSparse {
            indices_len: dist.len(),
            data_len: n,
        });
    }

    dist.fill(f64::INFINITY);
    dist[source] = 0.0;

    let mut heap: BinaryHeap<(RevOrderedFloat<f64>, u32)> = BinaryHeap::new();
    heap.push((RevOrderedFloat(0.0), source as u32));

    while let Some((d, u)) = heap.pop() {
        let d = d.get_value();
        let u = u as usize;
        // Lazy deletion: this entry was superseded before it reached the top.
        if d > dist[u] {
            continue;
        }
        for idx in graph.indptr[u] as usize..graph.indptr[u + 1] as usize {
            let v = graph.indices[idx] as usize;
            let w = graph.data[idx].to_f64().unwrap_or(f64::NAN);
            if !w.is_finite() {
                continue;
            }
            let candidate = d + w;
            if candidate < dist[v] {
                dist[v] = candidate;
                heap.push((RevOrderedFloat(candidate), v as u32));
            }
        }
    }

    Ok(())
}

/// Multi-source Dijkstra, one independent run per source.
///
/// Fans out over sources with rayon; each task owns one contiguous output row,
/// so there is no sharing and no reduction. Wall clock is therefore the slowest
/// single source rather than their sum.
///
/// The output is **row-major `sources.len() × n_nodes`**, which is what the
/// per-source runs write naturally. Callers wanting per-node reductions over
/// sources should block over nodes with the source loop inner rather than
/// transposing: that keeps every read contiguous and needs no second buffer,
/// which matters because this allocation is typically the largest in any
/// pipeline that uses it.
///
/// ### Params
///
/// * `graph` - Square CSR adjacency with non-negative weights. Non-finite
///   weights are skipped, as in [dijkstra_from_source].
/// * `sources` - Source node indices, one output row each.
///
/// ### Returns
///
/// Flat row-major `sources.len() × graph.nrows()` distances, `f64::INFINITY`
/// where a node is unreachable from that row's source.
pub fn multi_source_dijkstra<T>(
    graph: &CompressedSparseData2<T>,
    sources: &[usize],
) -> Result<Vec<f64>, BixverseErrors>
where
    T: BixverseFloat + Sync,
{
    let n = validate_graph(graph)?;

    if let Some(&bad) = sources.iter().find(|&&s| s >= n) {
        return Err(BixverseErrors::SliceIndexOutOfBounds { index: bad, len: n });
    }

    let mut out = vec![f64::INFINITY; sources.len() * n];

    out.par_chunks_mut(n)
        .zip(sources.par_iter())
        .try_for_each(|(row, &source)| dijkstra_from_source(graph, source, row))?;

    Ok(out)
}

/////////////
// Helpers //
/////////////

/// Check that a graph is a square CSR matrix and return its node count.
///
/// ### Params
///
/// * `graph` - The adjacency to validate.
///
/// ### Returns
///
/// The node count, or an error when the matrix is not CSR or not square.
fn validate_graph<T>(graph: &CompressedSparseData2<T>) -> Result<usize, BixverseErrors>
where
    T: BixverseFloat,
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
    use approx::assert_relative_eq;

    /// Build a symmetric CSR adjacency from an undirected edge list.
    ///
    /// ### Params
    ///
    /// * `n` - Node count.
    /// * `edges` - Undirected edges as `(from, to, weight)`.
    ///
    /// ### Returns
    ///
    /// The CSR adjacency with both directions stored.
    fn undirected_csr(n: usize, edges: &[(usize, usize, f64)]) -> CompressedSparseData2<f64> {
        let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        for &(a, b, w) in edges {
            rows[a].push((b as u32, w));
            rows[b].push((a as u32, w));
        }

        let mut indptr: Vec<u32> = Vec::with_capacity(n + 1);
        let mut indices: Vec<u32> = Vec::new();
        let mut data: Vec<f64> = Vec::new();
        indptr.push(0);
        for row in rows.iter_mut() {
            row.sort_by_key(|&(j, _)| j);
            for &(j, w) in row.iter() {
                indices.push(j);
                data.push(w);
            }
            indptr.push(indices.len() as u32);
        }

        CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (n, n))
    }

    /// Dense all-pairs reference via Floyd-Warshall.
    ///
    /// ### Params
    ///
    /// * `n` - Node count.
    /// * `edges` - Undirected edges as `(from, to, weight)`.
    ///
    /// ### Returns
    ///
    /// Flat row-major `n x n` distances.
    fn floyd_warshall(n: usize, edges: &[(usize, usize, f64)]) -> Vec<f64> {
        let mut d = vec![f64::INFINITY; n * n];
        for i in 0..n {
            d[i * n + i] = 0.0;
        }
        for &(a, b, w) in edges {
            d[a * n + b] = d[a * n + b].min(w);
            d[b * n + a] = d[b * n + a].min(w);
        }
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let via = d[i * n + k] + d[k * n + j];
                    if via < d[i * n + j] {
                        d[i * n + j] = via;
                    }
                }
            }
        }
        d
    }

    #[test]
    fn test_dijkstra_linear_chain() {
        let edges: Vec<(usize, usize, f64)> = (0..5).map(|i| (i, i + 1, 1.0)).collect();
        let graph = undirected_csr(6, &edges);

        let mut dist = vec![0.0; 6];
        dijkstra_from_source(&graph, 0, &mut dist).unwrap();

        for (i, &d) in dist.iter().enumerate() {
            assert_relative_eq!(d, i as f64, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_dijkstra_prefers_two_hop_shortcut() {
        // Direct 0-2 edge is heavier than routing through node 1.
        let edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 5.0)];
        let graph = undirected_csr(3, &edges);

        let mut dist = vec![0.0; 3];
        dijkstra_from_source(&graph, 0, &mut dist).unwrap();

        assert_relative_eq!(dist[2], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_dijkstra_unreachable_stays_infinite() {
        let edges = [(0, 1, 1.0), (2, 3, 1.0)];
        let graph = undirected_csr(4, &edges);

        let mut dist = vec![0.0; 4];
        dijkstra_from_source(&graph, 0, &mut dist).unwrap();

        assert!(dist[0].is_finite() && dist[1].is_finite());
        assert!(dist[2].is_infinite() && dist[3].is_infinite());
    }

    #[test]
    fn test_dijkstra_matches_floyd_warshall() {
        // Deterministic pseudo-random graph, no RNG dependency.
        let n = 20;
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n {
            for step in 1..4 {
                let j = (i * 7 + step * 3) % n;
                if i != j {
                    let w = 1.0 + ((i * 13 + step * 5) % 9) as f64;
                    edges.push((i, j, w));
                }
            }
        }

        let graph = undirected_csr(n, &edges);
        let reference = floyd_warshall(n, &edges);

        let sources: Vec<usize> = (0..n).collect();
        let got = multi_source_dijkstra(&graph, &sources).unwrap();

        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(got[i * n + j], reference[i * n + j], epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_multi_source_layout_is_row_major_per_source() {
        let edges: Vec<(usize, usize, f64)> = (0..4).map(|i| (i, i + 1, 2.0)).collect();
        let graph = undirected_csr(5, &edges);

        let got = multi_source_dijkstra(&graph, &[0, 4]).unwrap();

        // Row 0 is the walk away from node 0, row 1 the walk away from node 4.
        assert_relative_eq!(got[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(got[4], 8.0, epsilon = 1e-12);
        assert_relative_eq!(got[5], 8.0, epsilon = 1e-12);
        assert_relative_eq!(got[9], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_dijkstra_rejects_out_of_range_source() {
        let graph = undirected_csr(3, &[(0, 1, 1.0)]);
        let mut dist = vec![0.0; 3];

        assert!(dijkstra_from_source(&graph, 7, &mut dist).is_err());
        assert!(multi_source_dijkstra(&graph, &[0, 7]).is_err());
    }

    #[test]
    fn test_dijkstra_skips_non_finite_edges() {
        // 0 -1- 1 -NaN- 2 -1- 3, plus a heavy but finite 0-2 detour. Without an
        // explicit skip the NaN candidate never relaxes anything and node 2 is
        // only reachable through the detour, which is the silent-wrong-answer
        // case; with it, the routing is the same but for the right reason.
        let edges = [(0, 1, 1.0), (1, 2, f64::NAN), (2, 3, 1.0), (0, 2, 9.0)];
        let graph = undirected_csr(4, &edges);

        let mut dist = vec![0.0; 4];
        dijkstra_from_source(&graph, 0, &mut dist).unwrap();

        assert!(dist.iter().all(|d| !d.is_nan()));
        assert_relative_eq!(dist[1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(dist[2], 9.0, epsilon = 1e-12);
        assert_relative_eq!(dist[3], 10.0, epsilon = 1e-12);
    }

    #[test]
    fn test_dijkstra_leaves_a_nan_only_node_unreachable() {
        // The single edge onto node 1 is NaN, so node 1 has no usable route.
        let edges = [(0, 1, f64::NAN)];
        let graph = undirected_csr(2, &edges);

        let mut dist = vec![0.0; 2];
        dijkstra_from_source(&graph, 0, &mut dist).unwrap();

        assert_relative_eq!(dist[0], 0.0, epsilon = 1e-12);
        assert!(dist[1].is_infinite());
    }

    #[test]
    fn test_dijkstra_rejects_csc_input() {
        let graph = undirected_csr(3, &[(0, 1, 1.0)]).transform();
        let mut dist = vec![0.0; 3];

        assert!(dijkstra_from_source(&graph, 0, &mut dist).is_err());
    }
}
