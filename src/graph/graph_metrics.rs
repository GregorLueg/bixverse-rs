//! Metrics related to graphs - for now only conductance for community
//! membership.

use crate::prelude::*;

/// Compute conductance for each community.
///
/// Conductance of community C is `cut(C) / min(vol(C), vol(V \ C))` where
/// `cut(C)` is the total weight of edges with exactly one endpoint in C and
/// `vol(C)` is the sum of degrees of nodes in C. Lower = better isolated.
///
/// Communities with `vol(C) == 0` or `vol(V \ C) == 0` get NaN (undefined).
///
/// ### Params
///
/// * `graph` - Undirected sparse graph
/// * `communities` - Community label per node
///
/// ### Returns
///
/// Vector of conductance per community, indexed by community label.
pub fn conductance_undirected<T>(
    graph: &SparseGraph<T>,
    communities: &[usize],
) -> Result<Vec<T>, BixverseErrors>
where
    T: BixverseFloat + Clone + std::iter::Sum,
{
    if graph.is_directed() {
        return Err(BixverseErrors::GraphDirectedError);
    }
    if graph.get_node_number() != communities.len() {
        return Err(BixverseErrors::CommunityAssignmentMismatch);
    }

    let n = graph.get_node_number();
    if n == 0 {
        return Ok(Vec::new());
    }

    let n_comms = communities.iter().max().map(|&c| c + 1).unwrap_or(0);

    let mut volume = vec![T::zero(); n_comms];
    let mut cut = vec![T::zero(); n_comms];
    let mut total_volume = T::zero();

    for i in 0..n {
        let ci = communities[i];
        let (neighbours, weights) = graph.get_neighbours(i);

        for (&j, &w) in neighbours.iter().zip(weights.iter()) {
            volume[ci] += w;
            total_volume += w;
            if communities[j] != ci {
                cut[ci] += w;
            }
        }
    }

    let nan = T::from_f64(f64::NAN).unwrap();
    let mut out = Vec::with_capacity(n_comms);
    for c in 0..n_comms {
        let vol_c = volume[c];
        let vol_complement = total_volume - vol_c;
        let denom = if vol_c < vol_complement {
            vol_c
        } else {
            vol_complement
        };

        if denom == T::zero() {
            out.push(nan);
        } else {
            out.push(cut[c] / denom);
        }
    }

    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::sparse::coo_to_csr;

    fn build_undirected(n: usize, edges: &[(usize, usize, f64)]) -> SparseGraph<f64> {
        let mut rows = Vec::with_capacity(edges.len() * 2);
        let mut cols = Vec::with_capacity(edges.len() * 2);
        let mut vals = Vec::with_capacity(edges.len() * 2);
        for &(u, v, w) in edges {
            rows.push(u);
            cols.push(v);
            vals.push(w);
            rows.push(v);
            cols.push(u);
            vals.push(w);
        }
        let csr = coo_to_csr(&rows.index_cast(), &cols.index_cast(), &vals, (n, n));
        SparseGraph::new(n, csr, false)
    }

    fn build_directed(n: usize, edges: &[(usize, usize, f64)]) -> SparseGraph<f64> {
        let rows: Vec<usize> = edges.iter().map(|e| e.0).collect();
        let cols: Vec<usize> = edges.iter().map(|e| e.1).collect();
        let vals: Vec<f64> = edges.iter().map(|e| e.2).collect();
        let csr = coo_to_csr(&rows.index_cast(), &cols.index_cast(), &vals, (n, n));
        SparseGraph::new(n, csr, true)
    }

    #[test]
    fn two_disconnected_cliques_have_zero_conductance() {
        // Two triangles, no edges between them.
        let edges = [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (0, 2, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (3, 5, 1.0),
        ];
        let g = build_undirected(6, &edges);
        let comms = vec![0, 0, 0, 1, 1, 1];

        let cond = conductance_undirected(&g, &comms).unwrap();

        assert_eq!(cond.len(), 2);
        assert_eq!(cond[0], 0.0);
        assert_eq!(cond[1], 0.0);
    }

    #[test]
    fn single_bridge_edge() {
        // Two triangles connected by one edge of weight 1.
        // cut(C0) = 1, vol(C0) = 2+2+3 = 7 (node 2 has internal degree 2 + bridge 1)
        // vol(C1) = 7 by symmetry. min = 7. conductance = 1/7.
        let edges = [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (0, 2, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (3, 5, 1.0),
            (2, 3, 1.0),
        ];
        let g = build_undirected(6, &edges);
        let comms = vec![0, 0, 0, 1, 1, 1];

        let cond = conductance_undirected(&g, &comms).unwrap();

        let expected = 1.0 / 7.0;
        assert!((cond[0] - expected).abs() < 1e-12);
        assert!((cond[1] - expected).abs() < 1e-12);
    }

    #[test]
    fn weighted_edges_respected() {
        // K4 split 2/2. Internal edges weight 1, crossing edges weight 0.5.
        // For C0={0,1}: cut = 0.5*4 = 2.0 (each node has 2 crossing edges).
        // vol(C0) = internal(1)*2 + crossing(0.5)*4 = 2 + 2 = 4.
        // vol(C1) = 4. min = 4. conductance = 2/4 = 0.5.
        let edges = [
            (0, 1, 1.0),
            (2, 3, 1.0),
            (0, 2, 0.5),
            (0, 3, 0.5),
            (1, 2, 0.5),
            (1, 3, 0.5),
        ];
        let g = build_undirected(4, &edges);
        let comms = vec![0, 0, 1, 1];

        let cond = conductance_undirected(&g, &comms).unwrap();

        assert!((cond[0] - 0.5).abs() < 1e-12);
        assert!((cond[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn whole_graph_one_community_gives_nan() {
        let edges = [(0, 1, 1.0), (1, 2, 1.0)];
        let g = build_undirected(3, &edges);
        let comms = vec![0, 0, 0];

        let cond = conductance_undirected(&g, &comms).unwrap();

        assert_eq!(cond.len(), 1);
        assert!(cond[0].is_nan());
    }

    #[test]
    fn empty_graph_returns_empty() {
        let csr = coo_to_csr::<f64>(&[], &[], &[], (0, 0));
        let g = SparseGraph::new(0, csr, false);
        let cond = conductance_undirected(&g, &[]).unwrap();
        assert!(cond.is_empty());
    }

    #[test]
    fn directed_graph_errors() {
        let edges = [(0, 1, 1.0)];
        let g = build_directed(2, &edges);
        let result = conductance_undirected(&g, &[0, 0]);
        assert!(matches!(result, Err(BixverseErrors::GraphDirectedError)));
    }

    #[test]
    fn mismatched_lengths_error() {
        let edges = [(0, 1, 1.0)];
        let g = build_undirected(2, &edges);
        let result = conductance_undirected(&g, &[0, 0, 0]);
        assert!(matches!(
            result,
            Err(BixverseErrors::CommunityAssignmentMismatch)
        ));
    }
}
