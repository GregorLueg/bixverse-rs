//! End-to-end integration test for the metacells pipeline.
//!
//! Builds a synthetic dataset with K well-separated clusters of cells and
//! validates each pipeline stage against ground truth.
//!
//! ### Synthetic data
//!
//! * 200 cells, 4 clusters of 50, planted with ~20 marker genes per cluster.
//! * 500 genes total: ~80 markers (cluster-specific high expression),
//!   ~420 noise genes (low constant expression across all cells).
//! * Per-cell library size ~10 000 UMIs.
//!
//! ### Stage assertions
//!
//! 1. Downsample: library sizes capped at target; sparsity preserved.
//! 2. Select: selected genes substantially overlap planted markers.
//! 3. Similarity: within-cluster mean similarity > between-cluster + margin.
//! 4. KNN: rows L1-normalised; no self-loops; within-cluster edges dominate.
//! 5. Seeds: every cell assigned; high per-true-cluster purity.
#![allow(clippy::needless_range_loop)]
#![cfg(feature = "single-cell")]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Poisson};

use bixverse_rs::core::math::sparse::{
    CompressedSparseData2, CompressedSparseFormat, transpose_sparse,
};
use bixverse_rs::single_cell::sc_analysis::metacells2::{
    MC2KnnParams, Pile, SelectParams, SimilarityParams, build_knn_graph, choose_seeds,
    compute_similarity, downsample_pile, select_features,
};

///////////////////
// Test fixtures //
///////////////////

const N_CLUSTERS: usize = 4;
const CELLS_PER_CLUSTER: usize = 50;
const N_CELLS: usize = N_CLUSTERS * CELLS_PER_CLUSTER;
const MARKERS_PER_CLUSTER: usize = 20;
const NOISE_GENES: usize = 420;
const N_GENES: usize = N_CLUSTERS * MARKERS_PER_CLUSTER + NOISE_GENES;
const FIXTURE_SEED: u64 = 0xBADC0FFEE0DDF00D;

/// Synthetic ground-truth fixture.
struct Fixture {
    /// Cells × genes raw counts as CSR `Pile`.
    pile: Pile,
    /// True cluster id per cell (0..N_CLUSTERS).
    true_cluster: Vec<usize>,
    /// Set of gene indices that are markers (any cluster).
    marker_genes: Vec<usize>,
    /// Per-cluster gene marker indices, indexed `[cluster][k]`.
    cluster_markers: Vec<Vec<usize>>,
}

fn build_fixture() -> Fixture {
    let mut rng = StdRng::seed_from_u64(FIXTURE_SEED);

    // Marker layout: cluster k owns gene indices
    // [k*MARKERS_PER_CLUSTER, (k+1)*MARKERS_PER_CLUSTER).
    // Noise genes occupy the upper N_GENES range.
    let cluster_markers: Vec<Vec<usize>> = (0..N_CLUSTERS)
        .map(|k| (k * MARKERS_PER_CLUSTER..(k + 1) * MARKERS_PER_CLUSTER).collect())
        .collect();
    let marker_genes: Vec<usize> = (0..N_CLUSTERS * MARKERS_PER_CLUSTER).collect();

    let true_cluster: Vec<usize> = (0..N_CELLS).map(|i| i / CELLS_PER_CLUSTER).collect();

    // Per-gene Poisson lambdas — depend on cell's cluster.
    // Marker genes: high in own cluster, low elsewhere. We deliberately spread
    // each marker's "in-cluster" lambda across a wide range (5 .. 80) so that
    // markers do not all collapse into a single mean-rank window of the
    // relative-variance filter. Real biological markers exhibit similar
    // dispersion; collapsed-mean fixtures hide bugs in the windowed median.
    // Noise genes use a varied baseline (lambda 1 .. 5) for the same reason.
    let marker_in_lambdas: Vec<f64> =
        fix_lambdas(&mut rng, N_CLUSTERS * MARKERS_PER_CLUSTER, 5.0, 80.0);
    let marker_out_lambdas: Vec<f64> =
        fix_lambdas(&mut rng, N_CLUSTERS * MARKERS_PER_CLUSTER, 0.1, 1.0);
    let noise_lambdas: Vec<f64> = fix_lambdas(&mut rng, NOISE_GENES, 1.0, 5.0);

    // Build CSR row-by-row.
    let mut data: Vec<u32> = Vec::new();
    let mut indices: Vec<usize> = Vec::new();
    let mut indptr: Vec<usize> = Vec::with_capacity(N_CELLS + 1);
    indptr.push(0);

    for cell in 0..N_CELLS {
        let cluster = true_cluster[cell];
        for gene in 0..N_GENES {
            let lambda = if gene < N_CLUSTERS * MARKERS_PER_CLUSTER {
                let owner = gene / MARKERS_PER_CLUSTER;
                if owner == cluster {
                    marker_in_lambdas[gene]
                } else {
                    marker_out_lambdas[gene]
                }
            } else {
                noise_lambdas[gene - N_CLUSTERS * MARKERS_PER_CLUSTER]
            };
            let count = sample_poisson(&mut rng, lambda);
            if count > 0 {
                data.push(count);
                indices.push(gene);
            }
        }
        indptr.push(data.len());
    }

    let raw = CompressedSparseData2 {
        data,
        indices,
        indptr,
        cs_type: CompressedSparseFormat::Csr,
        data_2: None,
        shape: (N_CELLS, N_GENES),
    };

    let umis_per_cell: Vec<f32> = (0..N_CELLS)
        .map(|i| {
            let s: u64 = (raw.indptr[i]..raw.indptr[i + 1])
                .map(|idx| raw.data[idx] as u64)
                .sum();
            s as f32
        })
        .collect();

    let pile = Pile {
        cell_indices: (0..N_CELLS).collect(),
        raw,
        umis_per_cell,
        n_genes: N_GENES,
        downsampled: None,
        selected_gene_indices: None,
        selected_dense: None,
    };

    Fixture {
        pile,
        true_cluster,
        marker_genes,
        cluster_markers,
    }
}

fn sample_poisson(rng: &mut StdRng, lambda: f64) -> u32 {
    // Floor lambda to a small positive value to avoid the Poisson distribution
    // panicking on lambda == 0.
    let l = lambda.max(1e-6);
    let dist = Poisson::new(l).unwrap();
    dist.sample(rng) as u32
}

/// Generate `n` lambdas log-uniformly distributed in `[lo, hi]`. Log-uniform
/// gives a more biologically plausible spread than linear: many genes with
/// modest expression, a few with very high.
fn fix_lambdas(rng: &mut StdRng, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    (0..n)
        .map(|_| (log_lo + rng.random::<f64>() * (log_hi - log_lo)).exp())
        .collect()
}

/////////////
// Helpers //
/////////////

fn select_params() -> SelectParams {
    SelectParams {
        downsample_min_samples: 750,
        downsample_min_cell_quantile: 0.05,
        downsample_max_cell_quantile: 0.5,
        min_gene_total: Some(20),
        min_gene_top3: Some(3),
        min_gene_relative_variance: Some(0.1),
        min_genes: 20,
        relative_variance_window_size: 50,
        lateral_gene_mask: None,
    }
}

fn knn_params() -> MC2KnnParams {
    MC2KnnParams::default()
}

fn similarity_params() -> SimilarityParams {
    SimilarityParams::default()
}

///////////
// Tests //
///////////

#[test]
fn stage1_downsample_caps_libraries_and_preserves_sparsity() {
    let mut fix = build_fixture();
    let params = select_params();

    let raw_indices = fix.pile.raw.indices.clone();
    let raw_indptr = fix.pile.raw.indptr.clone();

    downsample_pile(&mut fix.pile, &params, FIXTURE_SEED);
    let down = fix
        .pile
        .downsampled
        .as_ref()
        .expect("downsampled populated");

    assert_eq!(down.indices, raw_indices);
    assert_eq!(down.indptr, raw_indptr);

    let mut sorted_umis = fix.pile.umis_per_cell.clone();
    sorted_umis.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted_umis[N_CELLS / 2];
    let target_upper = median.ceil() as u64 + 1;

    for i in 0..N_CELLS {
        let row_sum: u64 = (down.indptr[i]..down.indptr[i + 1])
            .map(|idx| down.data[idx] as u64)
            .sum();
        assert!(
            row_sum <= target_upper,
            "row {} downsampled to {} but target upper is {}",
            i,
            row_sum,
            target_upper
        );
    }
}

#[test]
fn stage2_select_recovers_marker_genes() {
    let mut fix = build_fixture();
    let params = select_params();

    downsample_pile(&mut fix.pile, &params, FIXTURE_SEED);
    select_features(&mut fix.pile, &params);

    let selected = fix
        .pile
        .selected_gene_indices
        .as_ref()
        .expect("selection populated");
    assert!(
        selected.len() >= params.min_genes,
        "selected only {} genes, expected >= {}",
        selected.len(),
        params.min_genes
    );

    let marker_set: std::collections::HashSet<usize> = fix.marker_genes.iter().copied().collect();
    let n_recovered_markers = selected.iter().filter(|g| marker_set.contains(g)).count();
    let recovery_rate = n_recovered_markers as f64 / selected.len() as f64;

    assert!(
        recovery_rate > 0.40,
        "marker recovery rate {:.3} below threshold (selected {}, markers {})",
        recovery_rate,
        selected.len(),
        n_recovered_markers
    );

    for (k, ms) in fix.cluster_markers.iter().enumerate() {
        let n_in_selection = ms.iter().filter(|g| selected.contains(g)).count();
        assert!(
            n_in_selection >= 1,
            "cluster {} has no markers in selection",
            k
        );
    }
}

#[test]
fn stage3_similarity_separates_clusters() {
    let mut fix = build_fixture();
    let params = select_params();

    downsample_pile(&mut fix.pile, &params, FIXTURE_SEED);
    select_features(&mut fix.pile, &params);
    let sim = compute_similarity(&fix.pile, &similarity_params()).unwrap();

    assert_eq!(sim.nrows(), N_CELLS);
    assert_eq!(sim.ncols(), N_CELLS);

    // Mean within-cluster vs between-cluster similarity (excluding diagonal).
    let mut sum_within = 0.0_f64;
    let mut count_within = 0usize;
    let mut sum_between = 0.0_f64;
    let mut count_between = 0usize;
    for i in 0..N_CELLS {
        for j in (i + 1)..N_CELLS {
            let s = sim[(i, j)] as f64;
            if fix.true_cluster[i] == fix.true_cluster[j] {
                sum_within += s;
                count_within += 1;
            } else {
                sum_between += s;
                count_between += 1;
            }
        }
    }
    let mean_within = sum_within / count_within as f64;
    let mean_between = sum_between / count_between as f64;
    let margin = mean_within - mean_between;

    assert!(
        margin > 0.2,
        "within-cluster mean {:.3} not sufficiently > between-cluster mean {:.3} (margin {:.3})",
        mean_within,
        mean_between,
        margin
    );
}

#[test]
fn stage4_knn_graph_is_normalised_and_clusters_dominate() {
    let mut fix = build_fixture();
    let params = select_params();

    downsample_pile(&mut fix.pile, &params, FIXTURE_SEED);
    select_features(&mut fix.pile, &params);
    let sim = compute_similarity(&fix.pile, &similarity_params()).unwrap();

    let k = 10;
    let graph = build_knn_graph(&sim, k, &knn_params());

    assert_eq!(graph.shape, (N_CELLS, N_CELLS));

    // Row L1 normalisation and no self-loops.
    for i in 0..N_CELLS {
        let start = graph.indptr[i];
        let end = graph.indptr[i + 1];
        let row_sum: f32 = graph.data[start..end].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-4 || row_sum == 0.0,
            "row {} sum is {}, expected 1.0",
            i,
            row_sum
        );
        for idx in start..end {
            assert!(
                graph.indices[idx] != i,
                "self-loop at row {} (idx {})",
                i,
                idx
            );
        }
    }

    // Within-cluster edges should dominate. Compute fraction of total edge
    // weight that stays within the true cluster.
    let mut weight_within = 0.0_f64;
    let mut weight_total = 0.0_f64;
    let mut count_within = 0usize;
    let mut count_total = 0usize;
    for i in 0..N_CELLS {
        let start = graph.indptr[i];
        let end = graph.indptr[i + 1];
        for idx in start..end {
            let j = graph.indices[idx];
            let w = graph.data[idx] as f64;
            weight_total += w;
            count_total += 1;
            if fix.true_cluster[i] == fix.true_cluster[j] {
                weight_within += w;
                count_within += 1;
            }
        }
    }
    let within_ratio = weight_within / weight_total;
    let count_ratio = count_within as f64 / count_total as f64;
    eprintln!(
        "stage4 diagnostics: edges total={} within={} count_ratio={:.3} weight_ratio={:.3}",
        count_total, count_within, count_ratio, within_ratio
    );

    // Also: for the pre-graph similarity, what fraction of each row's top-k
    // most similar entries (excluding self) are within-cluster? This isolates
    // whether the signal is in the similarity matrix vs being lost in the
    // graph construction.
    let mut sim_top_within = 0usize;
    let mut sim_top_total = 0usize;
    for i in 0..N_CELLS {
        let mut sims: Vec<(usize, f32)> = (0..N_CELLS)
            .filter(|&j| j != i)
            .map(|j| (j, sim[(i, j)]))
            .collect();
        sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for &(j, _) in sims.iter().take(k) {
            sim_top_total += 1;
            if fix.true_cluster[i] == fix.true_cluster[j] {
                sim_top_within += 1;
            }
        }
    }
    eprintln!(
        "stage4 diagnostics: similarity-top-{} within-fraction = {:.3}",
        k,
        sim_top_within as f64 / sim_top_total as f64
    );

    assert!(
        within_ratio > 0.85,
        "within-cluster weight ratio {:.3} below 0.85",
        within_ratio
    );
}

#[test]
fn stage5_seeds_assigned_with_high_purity() {
    let mut fix = build_fixture();
    let params = select_params();

    downsample_pile(&mut fix.pile, &params, FIXTURE_SEED);
    select_features(&mut fix.pile, &params);
    let sim = compute_similarity(&fix.pile, &similarity_params()).unwrap();
    let k = 10;
    let graph = build_knn_graph(&sim, k, &knn_params());

    // The seeds API takes the asymmetric outgoing graph plus an incoming
    // view (transpose, re-flagged as CSR semantically: row i = incoming
    // neighbours of cell i).
    let inc_t = transpose_sparse(&graph);
    let incoming = CompressedSparseData2 {
        data: inc_t.data,
        indices: inc_t.indices,
        indptr: inc_t.indptr,
        cs_type: CompressedSparseFormat::Csr,
        data_2: inc_t.data_2,
        shape: inc_t.shape,
    };

    // Ask for ~N_CLUSTERS seeds. Phase 3 will complete the assignment.
    let mut seeds = vec![-1i32; N_CELLS];
    let n_seeds = choose_seeds(
        &graph,
        &incoming,
        &mut seeds,
        N_CLUSTERS,
        0.0,
        1.0,
        FIXTURE_SEED,
    );

    // Every cell assigned, all ids in valid range.
    assert!(seeds.iter().all(|&s| s >= 0));
    assert!(seeds.iter().copied().all(|s| (s as usize) < n_seeds));

    // Per-true-cluster purity: for each true cluster, what fraction of its
    // cells share the most common seed assignment?
    let mut min_purity = 1.0_f64;
    for k in 0..N_CLUSTERS {
        let cluster_cells: Vec<usize> =
            (0..N_CELLS).filter(|&i| fix.true_cluster[i] == k).collect();
        let mut counts = vec![0usize; n_seeds];
        for &c in &cluster_cells {
            counts[seeds[c] as usize] += 1;
        }
        let dominant = *counts.iter().max().unwrap();
        let purity = dominant as f64 / cluster_cells.len() as f64;
        if purity < min_purity {
            min_purity = purity;
        }
    }

    assert!(
        min_purity > 0.7,
        "minimum per-cluster seed purity {:.3} below 0.7",
        min_purity
    );
}
