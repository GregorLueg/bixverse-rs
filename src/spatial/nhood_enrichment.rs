//! Neighbourhood enrichment on a spatial graph.
//!
//! Given a categorical label per spot (cluster id, annotated cell type) with
//! `K` levels and a spatial graph, measure whether each pair of labels appears
//! as neighbours more or less often than chance. Z-scores against a label
//! shuffle null.
//!
//! The graph is fixed; only the per-spot label assignment is permuted. Each
//! permutation rebuilds the K×K count matrix once and contributes to a
//! running mean / sum-of-squares. The final
//! `Z[a, b] = (observed[a, b] - mean[a, b]) / sd[a, b]`.
//!
//! Symmetrisation convention (applied both in observed and permutation
//! passes): each undirected edge contributes once to a diagonal entry
//! (`a == b`) or to both off-diagonal entries (`a != b → both N[a,b] and
//! N[b,a]`). This keeps the count matrix symmetric throughout. The optional
//! `params.symmetrise` then averages `Z` with `Zᵀ` at the very end (a no-op
//! for a symmetric count matrix, kept for parity with downstream R code).

use faer::Mat;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::graph::spatial_graph::GraphCsr;
use crate::prelude::*;

////////////
// Params //
////////////

/// Parameters for [`compute_nhood_enrichment`].
#[derive(Debug, Clone)]
pub struct NhoodEnrichmentParams {
    /// Number of permutations. Default 1000.
    pub n_permutations: usize,
    /// RNG seed for reproducibility. Permutation `k` uses
    /// `StdRng::seed_from_u64(seed.wrapping_add(k as u64))` so the result is
    /// independent of thread scheduling.
    pub seed: u64,
    /// Average `Z` with `Zᵀ` at the end. Default `true`. Has no effect on the
    /// numerical result while the count rule is the symmetric one above, but
    /// kept for explicit downstream control.
    pub symmetrise: bool,
}

impl Default for NhoodEnrichmentParams {
    fn default() -> Self {
        Self {
            n_permutations: 1000,
            seed: 0,
            symmetrise: true,
        }
    }
}

/////////////
// Results //
/////////////

/// Output of a neighbourhood-enrichment run.
#[derive(Debug, Clone)]
pub struct NhoodEnrichmentRes {
    /// Number of label levels.
    pub n_labels: usize,
    /// Original label names, in encoding order (`label_levels[i]` is the
    /// label whose code is `i`).
    pub label_levels: Vec<String>,
    /// K×K observed edge counts (symmetric).
    pub observed: Mat<f64>,
    /// K×K mean from permutations.
    pub perm_mean: Mat<f64>,
    /// K×K standard deviation from permutations.
    pub perm_sd: Mat<f64>,
    /// K×K Z-scores.
    pub z_scores: Mat<f64>,
}

//////////////
// Entry pt //
//////////////

/// Compute neighbourhood-enrichment Z-scores on `graph` with `labels`.
///
/// ### Params
///
/// * `graph` - Per-sample symmetric spatial graph.
/// * `labels` - One label code per node (`labels.len() == graph.n_nodes()`).
///   Codes must lie in `0..label_levels.len()`.
/// * `label_levels` - Human-readable name per code (kept in the result for
///   downstream display).
/// * `params` - Permutation count, seed, symmetrise toggle.
pub fn compute_nhood_enrichment(
    graph: &GraphCsr,
    labels: &[u32],
    label_levels: &[String],
    params: &NhoodEnrichmentParams,
) -> Result<NhoodEnrichmentRes, BixverseErrors> {
    let n_nodes = graph.n_nodes();
    if labels.len() != n_nodes {
        return Err(BixverseErrors::NhoodLabelLengthMismatch {
            n_labels: labels.len(),
            n_nodes,
        });
    }
    if params.n_permutations == 0 {
        return Err(BixverseErrors::NhoodZeroPermutations);
    }
    let k = label_levels.len();
    for &code in labels {
        if (code as usize) >= k {
            return Err(BixverseErrors::NhoodLabelOutOfRange {
                code,
                n_levels: k,
            });
        }
    }

    let edges: Vec<(u32, u32)> = graph.iter_edges_upper().collect();

    let observed_counts = count_edges_by_label(&edges, labels, k);

    // Permutation accumulators: sum and sum-of-squares per cell, flat (row-
    // major) K*K vectors. Permutation k uses a fresh deterministic RNG so the
    // final sums are independent of thread scheduling.
    let n_perm = params.n_permutations;
    let seed = params.seed;
    let init_labels = labels.to_vec();

    let (sum, sum_sq) = (0..n_perm)
        .into_par_iter()
        .fold(
            || PermAcc::new(k, &init_labels),
            |mut acc, perm_idx| {
                acc.run_one(perm_idx, seed, &edges, &init_labels);
                acc
            },
        )
        .reduce(
            || PermAcc::empty(k),
            |mut a, b| {
                a.merge_from(&b);
                a
            },
        )
        .into_running_totals();

    let n_perm_f = n_perm as f64;
    let mut observed = Mat::<f64>::zeros(k, k);
    let mut perm_mean = Mat::<f64>::zeros(k, k);
    let mut perm_sd = Mat::<f64>::zeros(k, k);
    let mut z_scores = Mat::<f64>::zeros(k, k);

    for a in 0..k {
        for b in 0..k {
            let idx = a * k + b;
            let obs = observed_counts[idx] as f64;
            observed[(a, b)] = obs;
            let mean = sum[idx] / n_perm_f;
            let var = (sum_sq[idx] / n_perm_f) - mean * mean;
            let var = if var < 0.0 { 0.0 } else { var };
            let sd = var.sqrt();
            perm_mean[(a, b)] = mean;
            perm_sd[(a, b)] = sd;
            z_scores[(a, b)] = if sd > 0.0 {
                (obs - mean) / sd
            } else {
                f64::NAN
            };
        }
    }

    if params.symmetrise {
        for a in 0..k {
            for b in (a + 1)..k {
                let avg = 0.5 * (z_scores[(a, b)] + z_scores[(b, a)]);
                z_scores[(a, b)] = avg;
                z_scores[(b, a)] = avg;
            }
        }
    }

    Ok(NhoodEnrichmentRes {
        n_labels: k,
        label_levels: label_levels.to_vec(),
        observed,
        perm_mean,
        perm_sd,
        z_scores,
    })
}

///////////////
// Internals //
///////////////

/// Symmetric K×K edge-count pass, flat row-major.
fn count_edges_by_label(edges: &[(u32, u32)], labels: &[u32], k: usize) -> Vec<u64> {
    let mut counts = vec![0_u64; k * k];
    for &(i, j) in edges {
        let a = labels[i as usize] as usize;
        let b = labels[j as usize] as usize;
        counts[a * k + b] += 1;
        if a != b {
            counts[b * k + a] += 1;
        }
    }
    counts
}

/// Per-thread permutation accumulator. Owns the shuffle scratch buffer and
/// the per-perm count buffer so they are reused across permutations within
/// the rayon fold.
struct PermAcc {
    k: usize,
    /// Reusable shuffle buffer; cloned from the original labels at init.
    shuffle: Vec<u32>,
    /// Reusable per-perm count matrix (cleared each iteration).
    counts: Vec<u64>,
    /// Running sum of per-cell counts across permutations.
    sum: Vec<f64>,
    /// Running sum of per-cell counts squared.
    sum_sq: Vec<f64>,
}

impl PermAcc {
    fn new(k: usize, init_labels: &[u32]) -> Self {
        Self {
            k,
            shuffle: init_labels.to_vec(),
            counts: vec![0_u64; k * k],
            sum: vec![0.0_f64; k * k],
            sum_sq: vec![0.0_f64; k * k],
        }
    }

    fn empty(k: usize) -> Self {
        Self {
            k,
            shuffle: Vec::new(),
            counts: Vec::new(),
            sum: vec![0.0_f64; k * k],
            sum_sq: vec![0.0_f64; k * k],
        }
    }

    fn run_one(
        &mut self,
        perm_idx: usize,
        seed: u64,
        edges: &[(u32, u32)],
        init_labels: &[u32],
    ) {
        let k = self.k;
        // Reset to the original assignment so the per-perm shuffle is a fresh
        // permutation of `init_labels` and independent of prior iterations.
        self.shuffle.copy_from_slice(init_labels);
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(perm_idx as u64));
        self.shuffle.shuffle(&mut rng);

        for c in self.counts.iter_mut() {
            *c = 0;
        }
        for &(i, j) in edges {
            let a = self.shuffle[i as usize] as usize;
            let b = self.shuffle[j as usize] as usize;
            self.counts[a * k + b] += 1;
            if a != b {
                self.counts[b * k + a] += 1;
            }
        }

        for idx in 0..(k * k) {
            let v = self.counts[idx] as f64;
            self.sum[idx] += v;
            self.sum_sq[idx] += v * v;
        }
    }

    fn merge_from(&mut self, other: &PermAcc) {
        debug_assert_eq!(self.sum.len(), other.sum.len());
        for idx in 0..self.sum.len() {
            self.sum[idx] += other.sum[idx];
            self.sum_sq[idx] += other.sum_sq[idx];
        }
    }

    fn into_running_totals(self) -> (Vec<f64>, Vec<f64>) {
        (self.sum, self.sum_sq)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::spatial_graph::make_weights_non_redundant;

    fn chain_graph(n: usize) -> GraphCsr {
        let mut neigh: Vec<Vec<usize>> = (0..n).map(|_| Vec::new()).collect();
        let mut w: Vec<Vec<f32>> = (0..n).map(|_| Vec::new()).collect();
        for i in 0..(n - 1) {
            neigh[i].push(i + 1);
            w[i].push(1.0);
            neigh[i + 1].push(i);
            w[i + 1].push(1.0);
        }
        let nr = make_weights_non_redundant(&neigh, &w);
        GraphCsr::from_non_redundant(&neigh, &nr)
    }

    #[test]
    fn observed_counts_are_symmetric() {
        let graph = chain_graph(6); // edges: (0,1),(1,2),(2,3),(3,4),(4,5)
        let labels = vec![0_u32, 0, 1, 1, 0, 0];
        let levels = vec!["A".to_string(), "B".to_string()];

        let res = compute_nhood_enrichment(
            &graph,
            &labels,
            &levels,
            &NhoodEnrichmentParams {
                n_permutations: 10,
                seed: 42,
                symmetrise: false,
            },
        )
        .unwrap();

        // edges: (0,1)=(A,A), (1,2)=(A,B), (2,3)=(B,B), (3,4)=(B,A), (4,5)=(A,A)
        // A,A diagonal: 2 within-A edges → 2
        // B,B diagonal: 1 within-B edge → 1
        // A,B + B,A: 2 cross edges, each contributing both → 2 each
        assert_eq!(res.observed[(0, 0)], 2.0);
        assert_eq!(res.observed[(1, 1)], 1.0);
        assert_eq!(res.observed[(0, 1)], 2.0);
        assert_eq!(res.observed[(1, 0)], 2.0);
    }

    #[test]
    fn permutation_z_is_reproducible() {
        let graph = chain_graph(10);
        let labels = vec![0_u32, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let levels = vec!["A".to_string(), "B".to_string()];

        let params = NhoodEnrichmentParams {
            n_permutations: 200,
            seed: 12345,
            symmetrise: true,
        };
        let r1 = compute_nhood_enrichment(&graph, &labels, &levels, &params).unwrap();
        let r2 = compute_nhood_enrichment(&graph, &labels, &levels, &params).unwrap();

        for a in 0..2 {
            for b in 0..2 {
                let z1 = r1.z_scores[(a, b)];
                let z2 = r2.z_scores[(a, b)];
                if z1.is_nan() || z2.is_nan() {
                    assert!(z1.is_nan() && z2.is_nan(), "NaN mismatch at ({a},{b})");
                } else {
                    assert!((z1 - z2).abs() < 1e-12, "Z mismatch at ({a},{b}): {z1} vs {z2}");
                }
            }
        }
    }

    #[test]
    fn clustered_labels_yield_positive_within_class_z() {
        // 30 nodes on a chain. Labels in 3 contiguous blocks of 10. Within-
        // class enrichment should be strongly positive vs. the shuffle null;
        // cross-class enrichment should be negative.
        let n = 30;
        let graph = chain_graph(n);
        let labels: Vec<u32> = (0..n as u32).map(|i| i / 10).collect();
        let levels = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let params = NhoodEnrichmentParams {
            n_permutations: 500,
            seed: 7,
            symmetrise: true,
        };
        let res = compute_nhood_enrichment(&graph, &labels, &levels, &params).unwrap();

        for a in 0..3 {
            assert!(
                res.z_scores[(a, a)] > 1.0,
                "expected positive within-class Z for class {a}, got {}",
                res.z_scores[(a, a)]
            );
        }
        // A-C never neighbour each other (separated by block B) so observed = 0
        // and mean > 0 → Z < 0.
        assert!(res.z_scores[(0, 2)] < 0.0, "{}", res.z_scores[(0, 2)]);
        assert!(res.z_scores[(2, 0)] < 0.0, "{}", res.z_scores[(2, 0)]);
    }

    #[test]
    fn mismatched_label_length_errors() {
        let graph = chain_graph(5);
        let labels = vec![0_u32, 1, 0]; // wrong length
        let levels = vec!["A".to_string(), "B".to_string()];

        let err = compute_nhood_enrichment(
            &graph,
            &labels,
            &levels,
            &NhoodEnrichmentParams::default(),
        )
        .unwrap_err();
        match err {
            BixverseErrors::NhoodLabelLengthMismatch { n_labels, n_nodes } => {
                assert_eq!(n_labels, 3);
                assert_eq!(n_nodes, 5);
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn out_of_range_label_errors() {
        let graph = chain_graph(4);
        let labels = vec![0_u32, 1, 2, 0]; // code 2 with only 2 levels
        let levels = vec!["A".to_string(), "B".to_string()];

        let err = compute_nhood_enrichment(
            &graph,
            &labels,
            &levels,
            &NhoodEnrichmentParams::default(),
        )
        .unwrap_err();
        match err {
            BixverseErrors::NhoodLabelOutOfRange { code, n_levels } => {
                assert_eq!(code, 2);
                assert_eq!(n_levels, 2);
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn zero_permutations_errors() {
        let graph = chain_graph(4);
        let labels = vec![0_u32, 1, 0, 1];
        let levels = vec!["A".to_string(), "B".to_string()];
        let params = NhoodEnrichmentParams {
            n_permutations: 0,
            seed: 0,
            symmetrise: true,
        };

        let err = compute_nhood_enrichment(&graph, &labels, &levels, &params).unwrap_err();
        assert!(matches!(err, BixverseErrors::NhoodZeroPermutations));
    }

    #[test]
    fn isolated_nodes_produce_zero_obs_and_nan_z() {
        // Graph with no edges → all observed counts and permutation counts
        // are zero → sd = 0 → Z is NaN.
        let graph = GraphCsr::from_non_redundant(&vec![vec![]; 3], &vec![vec![]; 3]);
        let labels = vec![0_u32, 0, 1];
        let levels = vec!["A".to_string(), "B".to_string()];

        let params = NhoodEnrichmentParams {
            n_permutations: 50,
            seed: 1,
            symmetrise: true,
        };
        let res = compute_nhood_enrichment(&graph, &labels, &levels, &params).unwrap();

        for a in 0..2 {
            for b in 0..2 {
                assert_eq!(res.observed[(a, b)], 0.0);
                assert_eq!(res.perm_mean[(a, b)], 0.0);
                assert!(res.z_scores[(a, b)].is_nan());
            }
        }
    }
}
