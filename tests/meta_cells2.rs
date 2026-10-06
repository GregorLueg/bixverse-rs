//! Stage-by-stage gates for the MetaCells2 pipeline on a planted-cluster
//! fixture.
//!
//! Four clusters of 50 cells, each with 20 Poisson marker genes, plus 420 noise
//! genes. Each test runs the pipeline up to one stage (downsampling, feature
//! selection, similarity, kNN graph, seeding, candidate metacells, deviants,
//! the direct end-to-end run) and checks that stage against the planted
//! truth. There is no external reference; the thresholds are recovery rates
//! the fixture should clear comfortably.

#![allow(clippy::needless_range_loop)]
#![cfg(feature = "single-cell")]

use std::collections::HashSet;

use rand::prelude::*;
use rand_distr::{Distribution, Poisson};

use bixverse_rs::core::math::sparse::transpose_sparse;
use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::mc_generation::metacells2::{
    DeviantsParams, DissolveParams, MC2KnnParams, MetacellsParams, PartitionParams, Pile,
    SelectParams, SimilarityParams, build_knn_graph, choose_seeds, compute_candidate_metacells,
    compute_direct_metacells, compute_similarity, downsample_pile, find_deviant_cells,
    make_incoming_view, select_features,
};

///////////////
// Constants //
///////////////

/// Planted clusters.
const N_CLUSTERS: usize = 4;
/// Cells per planted cluster.
const CELLS_PER_CLUSTER: usize = 50;
/// Total cells.
const N_CELLS: usize = N_CLUSTERS * CELLS_PER_CLUSTER;
/// Marker genes per cluster, high in their own cluster and low elsewhere.
const MARKERS_PER_CLUSTER: usize = 20;
/// Genes with no cluster structure.
const NOISE_GENES: usize = 420;
/// Total genes, markers first.
const N_GENES: usize = N_CLUSTERS * MARKERS_PER_CLUSTER + NOISE_GENES;
/// Seed for the fixture and for every seeded stage.
const FIXTURE_SEED: u64 = 0xBADC0FFEE0DDF00D;
/// Neighbours per cell in the kNN graph.
const KNN_K: usize = 10;

/////////////
// Fixture //
/////////////

/// Synthetic ground-truth fixture.
struct Fixture {
    /// Cells x genes raw counts as CSR `Pile`.
    pile: Pile,
    /// True cluster id per cell (0..N_CLUSTERS).
    true_cluster: Vec<usize>,
    /// Set of gene indices that are markers (any cluster).
    marker_genes: Vec<usize>,
    /// Per-cluster gene marker indices, indexed `[cluster][k]`.
    cluster_markers: Vec<Vec<usize>>,
}

/// Builds the planted-cluster fixture from [FIXTURE_SEED].
///
/// ### Returns
///
/// The [Fixture], with cells ordered by cluster.
fn build_fixture() -> Fixture {
    let mut rng = StdRng::seed_from_u64(FIXTURE_SEED);

    let cluster_markers: Vec<Vec<usize>> = (0..N_CLUSTERS)
        .map(|k| (k * MARKERS_PER_CLUSTER..(k + 1) * MARKERS_PER_CLUSTER).collect())
        .collect();
    let marker_genes: Vec<usize> = (0..N_CLUSTERS * MARKERS_PER_CLUSTER).collect();

    let true_cluster: Vec<usize> = (0..N_CELLS).map(|i| i / CELLS_PER_CLUSTER).collect();

    // Per-gene Poisson lambdas depend on the cell's cluster. Marker genes: high
    // in own cluster, low elsewhere. We deliberately spread each marker's
    // "in-cluster" lambda across a wide range (5 .. 80) so that markers do not
    // all collapse into a single mean-rank window of the relative-variance
    // filter. Real biological markers exhibit similar dispersion;
    // collapsed-mean fixtures hide bugs in the windowed median. Noise genes use
    // a varied baseline (lambda 1 .. 5) for the same reason.
    let marker_in_lambdas: Vec<f64> =
        fix_lambdas(&mut rng, N_CLUSTERS * MARKERS_PER_CLUSTER, 5.0, 80.0);
    let marker_out_lambdas: Vec<f64> =
        fix_lambdas(&mut rng, N_CLUSTERS * MARKERS_PER_CLUSTER, 0.1, 1.0);
    let noise_lambdas: Vec<f64> = fix_lambdas(&mut rng, NOISE_GENES, 1.0, 5.0);

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
        indices: indices.index_cast(),
        indptr: indptr.index_cast(),
        cs_type: CompressedSparseFormat::Csr,
        data_2: None,
        shape: (N_CELLS, N_GENES),
    };

    let umis_per_cell: Vec<f32> = (0..N_CELLS)
        .map(|i| {
            let s: u64 = (raw.indptr[i]..raw.indptr[i + 1])
                .map(|idx| raw.data[idx as usize] as u64)
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

/////////////
// Helpers //
/////////////

/// One Poisson draw, with `lambda` floored at `1e-6` so the distribution is
/// always valid.
///
/// ### Params
///
/// * `rng` - Source of randomness
/// * `lambda` - Poisson rate
///
/// ### Returns
///
/// One count.
fn sample_poisson(rng: &mut StdRng, lambda: f64) -> u32 {
    let l = lambda.max(1e-6);
    let dist = Poisson::new(l).expect("valid poisson");
    dist.sample(rng) as u32
}

/// Log-uniform rates in `[lo, hi)`.
///
/// ### Params
///
/// * `rng` - Source of randomness
/// * `n` - Number of rates
/// * `lo` - Lower bound
/// * `hi` - Upper bound
///
/// ### Returns
///
/// `n` rates.
fn fix_lambdas(rng: &mut StdRng, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    (0..n)
        .map(|_| (log_lo + rng.random::<f64>() * (log_hi - log_lo)).exp())
        .collect()
}

/// Fraction of `(cell_i, cell_j)` pairs within the same candidate metacell
/// that are also in the same true cluster.
///
/// ### Params
///
/// * `candidate_of_cell` - Candidate metacell per cell
/// * `true_cluster` - Planted cluster per cell
///
/// ### Returns
///
/// The fraction, 1.0 if candidates respect true cluster boundaries perfectly
/// (or no pair shares a candidate).
fn within_candidate_same_cluster_fraction(
    candidate_of_cell: &[i32],
    true_cluster: &[usize],
) -> f64 {
    let n = candidate_of_cell.len();
    let mut same = 0_u64;
    let mut total = 0_u64;
    for i in 0..n {
        for j in (i + 1)..n {
            if candidate_of_cell[i] == candidate_of_cell[j] {
                total += 1;
                if true_cluster[i] == true_cluster[j] {
                    same += 1;
                }
            }
        }
    }
    if total == 0 {
        1.0
    } else {
        same as f64 / total as f64
    }
}

////////////////
// Parameters //
////////////////

/// Feature selection parameters sized for the 200-cell fixture.
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

/// Default kNN parameters.
fn knn_params() -> MC2KnnParams {
    MC2KnnParams::default()
}

/// Default similarity parameters.
fn similarity_params() -> SimilarityParams {
    SimilarityParams::default()
}

/// Full pipeline parameters. Target metacell size 25, so several metacells per
/// planted cluster.
fn full_params() -> MetacellsParams {
    MetacellsParams {
        target_metacell_size: 25,
        target_metacell_umis: 50_000,
        min_metacell_size: 5,
        select: select_params(),
        similarity: SimilarityParams::default(),
        knn: MC2KnnParams {
            knn_k_override: Some(KNN_K),
            ..Default::default()
        },
        partition: PartitionParams {
            cooldown_pass: 0.05,
            cooldown_phase: 0.5,
            min_seed_size_quantile: 0.05,
            max_seed_size_quantile: 0.95,
            max_merge_size_factor: 0.25,
            ..Default::default()
        },
        deviants: DeviantsParams::default(),
        dissolve: DissolveParams {
            min_robust_size_factor: 0.25,
            min_convincing_gene_fold_factor: None,
        },
        must_complete_cover: false,
        random_seed: 42,
    }
}

///////////
// Tests //
///////////

/// Downsampling caps every library at the target and leaves the sparsity
/// pattern alone.
#[test]
fn test_stage1_downsample_caps_libraries_and_preserves_sparsity() {
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
    sorted_umis.sort_by(|a, b| a.partial_cmp(b).expect("finite UMI totals"));
    let median = sorted_umis[N_CELLS / 2];
    let target_upper = median.ceil() as u64 + 1;

    for i in 0..N_CELLS {
        let row_sum: u64 = (down.indptr[i]..down.indptr[i + 1])
            .map(|idx| down.data[idx as usize] as u64)
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

/// Feature selection is dominated by markers and leaves no cluster
/// unrepresented.
#[test]
fn test_stage2_select_recovers_marker_genes() {
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

    let marker_set: HashSet<usize> = fix.marker_genes.iter().copied().collect();
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

/// Within-cluster similarity has to beat between-cluster similarity by a clear
/// margin.
#[test]
fn test_stage3_similarity_separates_clusters() {
    let mut fix = build_fixture();
    let params = select_params();

    downsample_pile(&mut fix.pile, &params, FIXTURE_SEED);
    select_features(&mut fix.pile, &params);
    let sim = compute_similarity(&fix.pile, &similarity_params()).expect("similarity failed");

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

/// kNN rows are L1-normalised and self-loop free, with over 85% of the edge
/// weight within the true cluster.
#[test]
fn test_stage4_knn_graph_is_normalised_and_clusters_dominate() {
    let mut fix = build_fixture();
    let params = select_params();

    downsample_pile(&mut fix.pile, &params, FIXTURE_SEED);
    select_features(&mut fix.pile, &params);
    let sim = compute_similarity(&fix.pile, &similarity_params()).expect("similarity failed");

    let graph = build_knn_graph(&sim, KNN_K, &knn_params());

    assert_eq!(graph.shape, (N_CELLS, N_CELLS));

    for i in 0..N_CELLS {
        let start = graph.indptr[i] as usize;
        let end = graph.indptr[i + 1] as usize;
        let row_sum: f32 = graph.data[start..end].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-4 || row_sum == 0.0,
            "row {} sum is {}, expected 1.0",
            i,
            row_sum
        );
        for idx in start..end {
            assert!(
                graph.indices[idx] != i as u32,
                "self-loop at row {} (idx {})",
                i,
                idx
            );
        }
    }

    let mut weight_within = 0.0_f64;
    let mut weight_total = 0.0_f64;
    for i in 0..N_CELLS {
        let start = graph.indptr[i];
        let end = graph.indptr[i + 1];
        for idx in start..end {
            let idx = idx as usize;
            let j = graph.indices[idx];
            let w = graph.data[idx] as f64;
            weight_total += w;
            if fix.true_cluster[i] == fix.true_cluster[j as usize] {
                weight_within += w;
            }
        }
    }
    let within_ratio = weight_within / weight_total;

    assert!(
        within_ratio > 0.85,
        "within-cluster weight ratio {:.3} below 0.85",
        within_ratio
    );
}

/// Seeding leaves no cell unassigned and keeps each true cluster in one
/// dominant seed.
#[test]
fn test_stage5_seeds_assigned_with_high_purity() {
    let mut fix = build_fixture();
    let params = select_params();

    downsample_pile(&mut fix.pile, &params, FIXTURE_SEED);
    select_features(&mut fix.pile, &params);
    let sim = compute_similarity(&fix.pile, &similarity_params()).expect("similarity failed");
    let graph = build_knn_graph(&sim, KNN_K, &knn_params());

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
        let dominant = *counts.iter().max().expect("at least one seed");
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

/// Candidate metacells respect the true cluster boundaries instead of mixing
/// across them.
#[test]
fn test_stage6_candidate_metacells_recover_clusters() {
    let mut fix = build_fixture();
    let params = full_params();

    downsample_pile(&mut fix.pile, &params.select, FIXTURE_SEED);
    select_features(&mut fix.pile, &params.select);
    let sim = compute_similarity(&fix.pile, &params.similarity).expect("similarity failed");
    let outgoing = build_knn_graph(&sim, KNN_K, &params.knn);
    let incoming = make_incoming_view(&outgoing);

    let candidates = compute_candidate_metacells(
        &outgoing,
        &incoming,
        &fix.pile.umis_per_cell,
        &params,
        FIXTURE_SEED,
    );

    assert!(candidates.iter().all(|&c| c >= 0));
    let n_candidates = (*candidates.iter().max().expect("non-empty candidates") + 1) as usize;
    assert!(
        n_candidates >= N_CLUSTERS,
        "expected >= {} candidate metacells, got {}",
        N_CLUSTERS,
        n_candidates
    );

    // Pair purity rather than one-candidate-per-cluster: how many metacells
    // each true cluster splits into is a function of the size budget, not of
    // correctness.
    let pair_purity = within_candidate_same_cluster_fraction(&candidates, &fix.true_cluster);

    assert!(
        pair_purity > 0.95,
        "within-candidate pair purity {:.3} is below 0.95, candidates are mixing true clusters",
        pair_purity
    );
}

/// Deviant detection stays near its configured ceiling on data with no real outliers.
#[test]
fn test_stage7_deviants_dont_flag_well_behaved_cells() {
    let mut fix = build_fixture();
    let params = full_params();

    downsample_pile(&mut fix.pile, &params.select, FIXTURE_SEED);
    select_features(&mut fix.pile, &params.select);
    let sim = compute_similarity(&fix.pile, &params.similarity).expect("similarity failed");
    let outgoing = build_knn_graph(&sim, KNN_K, &params.knn);
    let incoming = make_incoming_view(&outgoing);
    let candidates = compute_candidate_metacells(
        &outgoing,
        &incoming,
        &fix.pile.umis_per_cell,
        &params,
        FIXTURE_SEED,
    );

    let deviants = find_deviant_cells(
        &fix.pile.raw,
        &fix.pile.umis_per_cell,
        &candidates,
        &params.deviants,
    );

    let n_deviants = deviants.iter().filter(|&&d| d).count();
    let frac = n_deviants as f64 / N_CELLS as f64;

    // Synthetic data has no genuine outliers. Some cells may still be
    // flagged due to Poisson tail noise, so bound the fraction loosely.
    assert!(
        frac < params.deviants.max_cell_fraction as f64 + 0.05,
        "deviant fraction {:.3} exceeds expected ceiling",
        frac
    );
}

/// End to end, the outlier flags stay consistent, the metacell IDs stay dense,
/// and at least 70% of each true cluster lands in some metacell.
#[test]
fn test_stage8_direct_pipeline_produces_valid_output() {
    let mut fix = build_fixture();
    let params = full_params();

    let result = compute_direct_metacells(&mut fix.pile, &params, FIXTURE_SEED as usize, true)
        .expect("direct metacells failed");

    assert_eq!(result.metacell_of_cell.len(), N_CELLS);
    assert_eq!(result.deviant_of_cell.len(), N_CELLS);
    assert_eq!(result.dissolved_of_cell.len(), N_CELLS);

    // A cell is never both deviant and dissolved.
    for i in 0..N_CELLS {
        assert!(
            !(result.deviant_of_cell[i] && result.dissolved_of_cell[i]),
            "cell {} is both deviant and dissolved",
            i
        );
    }

    // Metacell == -1 iff deviant or dissolved.
    for i in 0..N_CELLS {
        let is_outlier = result.deviant_of_cell[i] || result.dissolved_of_cell[i];
        assert_eq!(
            result.metacell_of_cell[i] < 0,
            is_outlier,
            "cell {} metacell={} but outlier={}",
            i,
            result.metacell_of_cell[i],
            is_outlier
        );
    }

    // Metacell IDs dense in [0, n_metacells).
    let max_id = *result
        .metacell_of_cell
        .iter()
        .max()
        .expect("non-empty assignment");
    if result.n_metacells > 0 {
        assert_eq!(max_id as usize, result.n_metacells - 1);
    }

    assert!(result.n_metacells >= N_CLUSTERS);
    for k in 0..N_CLUSTERS {
        let cluster_cells: Vec<usize> =
            (0..N_CELLS).filter(|&i| fix.true_cluster[i] == k).collect();
        let n_assigned = cluster_cells
            .iter()
            .filter(|&&i| result.metacell_of_cell[i] >= 0)
            .count();
        let assigned_frac = n_assigned as f64 / cluster_cells.len() as f64;
        assert!(
            assigned_frac >= 0.7,
            "cluster {} only {:.1}% assigned",
            k,
            100.0 * assigned_frac
        );
    }
}
