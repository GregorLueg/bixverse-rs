//! Symmetric CSR graph used by spatial autocorrelation methods (HotSpot,
//! Moran's I, neighbourhood enrichment, SuperSpot).
//!
//! The graph is built from per-node neighbour/weight adjacency lists in two
//! steps:
//!
//! 1. [`make_weights_non_redundant`] collapses each undirected edge to a
//!    single canonical entry (combined weight on the lower-index endpoint,
//!    zero on the reciprocal entry).
//! 2. [`GraphCsr::from_non_redundant`] scatters those canonical entries into
//!    both rows, producing a symmetric CSR matrix.
//!
//! Downstream code consumes the symmetric CSR. Convenience helpers cover the
//! operations used across the spatial methods:
//!
//! - [`GraphCsr::spmv`] / [`GraphCsr::spmv_sq`] - sparse mat-vec against
//!   `W_sym` and `W_sym²`.
//! - [`GraphCsr::quadratic_form`] - fused `cᵀ W_sym c` for Moran's I and the
//!   HotSpot pair statistic.
//! - [`GraphCsr::iter_edges_upper`] - undirected edge iterator (each pair
//!   yielded once with `i < j`), used by neighbourhood enrichment.
//! - [`compute_node_degree`] - sum of incident weights per node.

///////////////////
// Graph struct  //
///////////////////

/// Symmetric graph in CSR layout shared across spatial methods.
///
/// Built from the non-redundant (upper-triangular combined) weights produced
/// by [`make_weights_non_redundant`]. Each undirected edge is stored in *both*
/// rows with its combined weight, so a single sparse mat-vec `wy = W_sym @ c`
/// reproduces exactly the symmetric neighbour-weighted vector that a hand
/// scatter would build. With that:
///
/// - `lc(x, y) = dot(x, W_sym @ y)` (HotSpot pair statistic, no extra factor)
/// - `eg2(x)   = sum_squares(W_sym @ x)`
/// - Moran's I numerator = `quadratic_form(x_centred)`
#[derive(Clone, Debug)]
pub struct GraphCsr {
    /// Row offsets, length `n_nodes + 1`.
    offsets: Vec<usize>,
    /// Column indices (the neighbour node), compacted (no zero-weight entries).
    indices: Vec<u32>,
    /// Edge weights, aligned with `indices`.
    weights: Vec<f32>,
}

impl GraphCsr {
    /// Build the symmetric CSR from the non-redundant neighbour/weight arrays.
    ///
    /// Iterates the non-zero (canonical) entries of `weights` and scatters
    /// each into both endpoints' rows. Zero-weight reciprocal entries are
    /// dropped.
    ///
    /// ### Params
    ///
    /// * `neighbours` - Neighbour indices for each node
    /// * `weights` - Non-redundant edge weights for each neighbour connection
    ///
    /// ### Returns
    ///
    /// A new `GraphCsr` with symmetric edges in CSR layout.
    pub fn from_non_redundant(neighbours: &[Vec<usize>], weights: &[Vec<f32>]) -> Self {
        let n = neighbours.len();
        let mut rows: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];

        for i in 0..n {
            for (k, &j) in neighbours[i].iter().enumerate() {
                let w = weights[i][k];
                if w == 0.0 {
                    continue;
                }
                rows[i].push((j as u32, w));
                rows[j].push((i as u32, w));
            }
        }

        let mut offsets = Vec::with_capacity(n + 1);
        let mut indices = Vec::new();
        let mut ws = Vec::new();
        offsets.push(0);
        for row in &rows {
            for &(j, w) in row {
                indices.push(j);
                ws.push(w);
            }
            offsets.push(indices.len());
        }

        Self {
            offsets,
            indices,
            weights: ws,
        }
    }

    /// Number of nodes in the graph.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Number of (directed) entries in the CSR. Each undirected edge with
    /// non-zero weight contributes 2 entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Read-only access to the row offsets (length `n_nodes + 1`).
    #[inline]
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Read-only access to the column indices.
    #[inline]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Read-only access to the edge weights.
    #[inline]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Sum of all stored edge weights. For a symmetric CSR this is `2 * S0`
    /// in Cliff-Ord notation (each undirected edge counted twice).
    pub fn weight_sum(&self) -> f64 {
        let mut s = 0.0_f64;
        for &w in &self.weights {
            s += w as f64;
        }
        s
    }

    /// Sparse matrix-vector product: `out = W_sym @ c`.
    ///
    /// ### Params
    ///
    /// * `c` - Input vector of length `n_nodes`
    /// * `out` - Output vector of length `n_nodes`, written in place
    pub fn spmv(&self, c: &[f32], out: &mut [f32]) {
        for i in 0..self.n_nodes() {
            let start = self.offsets[i];
            let end = self.offsets[i + 1];
            let mut acc = 0.0_f32;
            for k in start..end {
                acc += self.weights[k] * c[self.indices[k] as usize];
            }
            out[i] = acc;
        }
    }

    /// Quadratic form `cᵀ W_sym c`, fused (no intermediate `W_sym @ c`).
    ///
    /// For Moran's I, the (un-normalised) numerator is exactly
    /// `quadratic_form(x_centred)`. For HotSpot's autocorrelation,
    /// `g = 0.5 * quadratic_form(vals)` because `W_sym` carries each
    /// undirected edge in both rows.
    ///
    /// ### Params
    ///
    /// * `c` - Input vector of length `n_nodes`
    ///
    /// ### Returns
    ///
    /// The scalar `cᵀ W_sym c`.
    pub fn quadratic_form(&self, c: &[f32]) -> f32 {
        let mut total = 0.0_f32;
        for i in 0..self.n_nodes() {
            let start = self.offsets[i];
            let end = self.offsets[i + 1];
            let mut acc = 0.0_f32;
            for k in start..end {
                acc += self.weights[k] * c[self.indices[k] as usize];
            }
            total += c[i] * acc;
        }
        total
    }

    /// Same as [`quadratic_form`] but in `f64`. Use when the centred values
    /// are already promoted to `f64` (Moran's I path keeps the gene chunk in
    /// `f32` and only the reduction needs to be in `f64`; this helper is for
    /// callers who promote earlier).
    pub fn quadratic_form_f64(&self, c: &[f64]) -> f64 {
        let mut total = 0.0_f64;
        for i in 0..self.n_nodes() {
            let start = self.offsets[i];
            let end = self.offsets[i + 1];
            let mut acc = 0.0_f64;
            for k in start..end {
                acc += (self.weights[k] as f64) * c[self.indices[k] as usize];
            }
            total += c[i] * acc;
        }
        total
    }

    /// Sparse mat-vec against the squared weights: `out = W_sym² @ c`.
    ///
    /// Same traversal as [`GraphCsr::spmv`] with each weight squared.
    ///
    /// ### Params
    ///
    /// * `c` - Input vector of length `n_nodes`
    /// * `out` - Output vector of length `n_nodes`, written in place
    pub fn spmv_sq(&self, c: &[f32], out: &mut [f32]) {
        for i in 0..self.n_nodes() {
            let start = self.offsets[i];
            let end = self.offsets[i + 1];
            let mut acc = 0.0_f32;
            for k in start..end {
                let w = self.weights[k];
                acc += w * w * c[self.indices[k] as usize];
            }
            out[i] = acc;
        }
    }

    /// Iterator over undirected edges with `i < j`. Each pair is yielded
    /// exactly once. Useful for symmetric statistics (neighbourhood
    /// enrichment, edge-count moments).
    pub fn iter_edges_upper(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        (0..self.n_nodes() as u32).flat_map(move |i| {
            let start = self.offsets[i as usize];
            let end = self.offsets[i as usize + 1];
            self.indices[start..end]
                .iter()
                .filter_map(move |&j| if j > i { Some((i, j)) } else { None })
        })
    }
}

////////////////////
// Free helpers   //
////////////////////

/// Compute Cliff-Ord-style graph weight sums used by Moran's I analytical
/// variance.
///
/// Returns `(s0, s1, s2)` where (definitions from Cliff & Ord 1981):
/// - `s0 = sum_ij w_ij`
/// - `s1 = 0.5 * sum_ij (w_ij + w_ji)^2`
/// - `s2 = sum_i (sum_j w_ij + sum_j w_ji)^2`
///
/// All sums are taken over the doubly-stored symmetric CSR. For symmetric W
/// (which this CSR always is), `w_ij == w_ji` so each ordered (i,j) entry
/// contributes `(2 w)^2 = 4 w^2` to the un-halved s1 sum; with both directions
/// stored that is `4 w^2` per directed entry, and `s1` ends up as
/// `2 * sum_directed w^2` (equivalently `4 * sum_upper w^2`).
pub fn graph_weight_moments(graph: &GraphCsr) -> (f64, f64, f64) {
    let weights = graph.weights();
    let offsets = graph.offsets();
    let indices = graph.indices();
    let n = graph.n_nodes();

    let mut s0 = 0.0_f64;
    let mut s1_acc = 0.0_f64;
    let mut row_sum = vec![0.0_f64; n];
    let mut col_sum = vec![0.0_f64; n];

    for i in 0..n {
        for k in offsets[i]..offsets[i + 1] {
            let w = weights[k] as f64;
            let j = indices[k] as usize;
            s0 += w;
            row_sum[i] += w;
            col_sum[j] += w;
            // per directed entry (i,j) of a symmetric W, the matching
            // summand of (w_ij + w_ji)^2 is (2w)^2 = 4 w^2.
            s1_acc += 4.0 * w * w;
        }
    }
    let s1 = 0.5 * s1_acc;

    let mut s2 = 0.0_f64;
    for i in 0..n {
        let rc = row_sum[i] + col_sum[i];
        s2 += rc * rc;
    }
    (s0, s1, s2)
}

/// Collapse reciprocal neighbour entries into a single canonical weight per
/// undirected edge.
///
/// For each pair `(i, j)` with `i < j` and both directions present, the
/// combined weight `w_ij + w_ji` is written into node `i`'s entry and node
/// `j`'s reciprocal entry is zeroed.
///
/// ### Params
///
/// * `neighbours` - Neighbour indices for each node
/// * `weights` - Edge weights for each neighbour connection
///
/// ### Returns
///
/// Modified weights with redundant edges zeroed.
pub fn make_weights_non_redundant(
    neighbours: &[Vec<usize>],
    weights: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let mut w_no_redundant = weights.to_vec();

    for i in 0..neighbours.len() {
        for k in 0..neighbours[i].len() {
            let j = neighbours[i][k];

            if j < i {
                continue;
            }

            for k2 in 0..neighbours[j].len() {
                if neighbours[j][k2] == i {
                    let w_ji = w_no_redundant[j][k2];
                    w_no_redundant[j][k2] = 0.0;
                    w_no_redundant[i][k] += w_ji;
                    break;
                }
            }
        }
    }

    w_no_redundant
}

/// Compute node degree from edge weights.
///
/// Each edge contributes to the degree of both its endpoints.
///
/// ### Params
///
/// * `neighbours` - Neighbour indices for each node
/// * `weights` - Edge weights for each neighbour connection (the non-redundant
///   representation produced by [`make_weights_non_redundant`])
///
/// ### Returns
///
/// Vector of degree values for each node.
pub fn compute_node_degree(neighbours: &[Vec<usize>], weights: &[Vec<f32>]) -> Vec<f32> {
    let mut d = vec![0.0_f32; neighbours.len()];

    for i in 0..neighbours.len() {
        for k in 0..neighbours[i].len() {
            let j = neighbours[i][k];
            let w_ij = weights[i][k];

            d[i] += w_ij;
            d[j] += w_ij;
        }
    }

    d
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn small_graph() -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let neighbours = vec![
            vec![1, 2],    // 0
            vec![0, 2],    // 1
            vec![0, 1, 3], // 2
            vec![2],       // 3
        ];
        let weights = vec![
            vec![0.5, 0.25],
            vec![0.5, 0.75],
            vec![0.25, 0.75, 1.0],
            vec![1.0],
        ];
        (neighbours, weights)
    }

    #[test]
    fn non_redundant_combines_and_zeroes() {
        let (neigh, w) = small_graph();
        let nr = make_weights_non_redundant(&neigh, &w);

        assert_eq!(nr[0][0], 1.0); // 0 -> 1
        assert_eq!(nr[1][0], 0.0);
        assert_eq!(nr[0][1], 0.5); // 0 -> 2
        assert_eq!(nr[2][0], 0.0);
        assert_eq!(nr[1][1], 1.5); // 1 -> 2
        assert_eq!(nr[2][1], 0.0);
        assert_eq!(nr[2][2], 2.0); // 2 -> 3
        assert_eq!(nr[3][0], 0.0);
    }

    #[test]
    fn node_degree_counts_both_endpoints() {
        let (neigh, w) = small_graph();
        let nr = make_weights_non_redundant(&neigh, &w);
        let d = compute_node_degree(&neigh, &nr);

        assert!((d[0] - 1.5).abs() < 1e-6);
        assert!((d[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn spmv_matches_hand_scatter() {
        let (neigh, w) = small_graph();
        let nr = make_weights_non_redundant(&neigh, &w);
        let graph = GraphCsr::from_non_redundant(&neigh, &nr);

        let x = [1.0_f32, -2.0, 3.0, 0.5];
        let mut wy = vec![0.0_f32; x.len()];
        graph.spmv(&x, &mut wy);

        let mut t = vec![0.0_f32; x.len()];
        for i in 0..neigh.len() {
            for k in 0..neigh[i].len() {
                let j = neigh[i][k];
                let wij = nr[i][k];
                if wij == 0.0 {
                    continue;
                }
                t[i] += wij * x[j];
                t[j] += wij * x[i];
            }
        }

        for (a, b) in wy.iter().zip(t.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn quadratic_form_matches_dot_with_spmv() {
        let (neigh, w) = small_graph();
        let nr = make_weights_non_redundant(&neigh, &w);
        let graph = GraphCsr::from_non_redundant(&neigh, &nr);

        let x = [1.0_f32, 0.5, -1.5, 2.0];
        let mut wx = vec![0.0_f32; x.len()];
        graph.spmv(&x, &mut wx);

        let dot: f32 = x.iter().zip(wx.iter()).map(|(&a, &b)| a * b).sum();
        let q = graph.quadratic_form(&x);
        assert!((q - dot).abs() < 1e-5, "{q} vs {dot}");
    }

    #[test]
    fn iter_edges_upper_yields_each_pair_once() {
        let (neigh, w) = small_graph();
        let nr = make_weights_non_redundant(&neigh, &w);
        let graph = GraphCsr::from_non_redundant(&neigh, &nr);

        let mut edges: Vec<(u32, u32)> = graph.iter_edges_upper().collect();
        edges.sort();
        assert_eq!(edges, vec![(0, 1), (0, 2), (1, 2), (2, 3)]);
    }

    #[test]
    fn weight_sum_double_counts_symmetric_edges() {
        let (neigh, w) = small_graph();
        let nr = make_weights_non_redundant(&neigh, &w);
        let graph = GraphCsr::from_non_redundant(&neigh, &nr);

        // upper-triangular weights: 1.0 + 0.5 + 1.5 + 2.0 = 5.0
        // symmetric storage doubles this: 10.0
        assert!((graph.weight_sum() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn graph_weight_moments_match_definition() {
        let (neigh, w) = small_graph();
        let nr = make_weights_non_redundant(&neigh, &w);
        let graph = GraphCsr::from_non_redundant(&neigh, &nr);

        let (s0, s1, s2) = graph_weight_moments(&graph);

        // s0 = sum of doubly-stored weights = 2 * sum_upper = 10.0
        assert!((s0 - 10.0).abs() < 1e-6);
        // s1 = 0.5 * sum_ij (w_ij + w_ji)^2 = 0.5 * 8 * sum_upper w^2
        //    = 4 * sum_upper w^2 (upper w^2 = 1 + 0.25 + 2.25 + 4 = 7.5 → 30.0)
        assert!((s1 - 30.0).abs() < 1e-6);
        // s2: for symmetric weights row_sum == col_sum, so s2 = sum_i (2*d_i)^2
        // d = [1.5, 2.5, 4.0, 2.0], 2d = [3, 5, 8, 4], sq = [9, 25, 64, 16] = 114
        assert!((s2 - 114.0).abs() < 1e-6);
    }
}
