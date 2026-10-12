//! End-to-end parity between the CPU and MLX SEACells paths.
//!
//! Only the B-gradient argmin differs between the two paths, so the
//! trajectories should track each other closely. Near-ties can still resolve
//! differently, so the assertions are on agreement rate and the objective
//! rather than on exact equality.
//!
//! n = 3000 over two pruning arms, each running a full CPU fit and a full MLX
//! fit, so the whole file is behind `large-test`.

#![cfg(all(feature = "single-cell", mlx_available, feature = "large-test"))]

use faer::Mat;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

use bixverse_rs::mlx::sc::seacells::seacells_fit_mlx;
use bixverse_rs::prelude::LanczosParams;
use bixverse_rs::single_cell::mc_generation::seacells::{SEACells, SEACellsParams};
use bixverse_rs::single_cell::sc_processing::knn::{KnnParams, generate_knn_with_dist};

///////////////
// Constants //
///////////////

/// Number of cells.
const N_CELLS: usize = 3000;
/// Number of SEACells.
const N_SEACELLS: usize = 40;
/// Embedding width.
const EMBED_DIM: usize = 20;
/// Number of blobs in the synthetic embedding.
const N_CLUSTERS: usize = 12;
/// Seed for the synthetic embedding.
const EMBED_SEED: u64 = 7;
/// Seed for the kNN graph, archetype initialisation and both fits.
const SEED: usize = 42;

/////////////
// Helpers //
/////////////

/// Clustered synthetic embedding.
///
/// ### Params
///
/// * `n` - Number of cells
/// * `dim` - Embedding width
/// * `n_clusters` - Number of blobs
/// * `seed` - Random seed
///
/// ### Returns
///
/// An `n x dim` matrix.
fn make_embedding(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Mat<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centre_dist = Normal::new(0.0f32, 1.0).expect("valid normal");
    let noise_dist = Normal::new(0.0f32, 0.5).expect("valid normal");

    let centres: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| centre_dist.sample(&mut rng)).collect())
        .collect();
    let assignments: Vec<usize> = (0..n).map(|_| rng.random_range(0..n_clusters)).collect();

    Mat::from_fn(n, dim, |i, j| {
        centres[assignments[i]][j] + noise_dist.sample(&mut rng)
    })
}

/// Parameters for the parity runs.
///
/// ### Params
///
/// * `k` - Number of SEACells
/// * `pruning` - Whether to prune each Frank-Wolfe iteration
///
/// ### Returns
///
/// The populated [SEACellsParams].
fn params(k: usize, pruning: bool) -> SEACellsParams {
    let mut knn_params = KnnParams::new();
    knn_params.k = 15;
    knn_params.ann_dist = "euclidean".to_string();

    SEACellsParams {
        n_sea_cells: k,
        max_fw_iters: 30,
        convergence_epsilon: 1e-3,
        max_iter: 2,
        min_iter: 2,
        greedy_threshold: 0,
        graph_building: "union".to_string(),
        pruning,
        pruning_threshold: 1e-7,
        n_landmarks: None,
        knn_params,
        lanczos_params: LanczosParams::default(),
    }
}

///////////
// Tests //
///////////

/// The MLX path must reach the same solution as the CPU path, with pruning
/// both off and on.
#[test]
fn test_seacells_mlx_matches_cpu() {
    let n = N_CELLS;
    let embedding = make_embedding(n, EMBED_DIM, N_CLUSTERS, EMBED_SEED);

    for pruning in [false, true] {
        let p = params(N_SEACELLS, pruning);

        let (knn_indices, knn_distances) =
            generate_knn_with_dist(embedding.as_ref(), &p.knn_params, true, false, SEED, false)
                .expect("kNN failed");
        let knn_distances = knn_distances.expect("distances requested");

        let mut cpu_model = SEACells::new(n, &p);
        cpu_model.construct_kernel_mat(embedding.as_ref(), &knn_indices, &knn_distances, 0);
        cpu_model
            .initialise_archetypes(&knn_indices, &knn_distances, 0, SEED as u64)
            .expect("archetype init failed");
        cpu_model.fit(SEED, 0).expect("CPU fit failed");
        let cpu_assign = cpu_model
            .get_hard_assignments()
            .expect("assignments failed");
        let cpu_rss = cpu_model.get_rss_history().to_vec();

        let (mlx_assign, mlx_metacells, _, mlx_rss) = seacells_fit_mlx(
            embedding.as_ref(),
            &knn_indices,
            &knn_distances,
            &p,
            SEED,
            0,
        )
        .expect("MLX fit failed");

        assert_eq!(mlx_assign.len(), n);
        assert!(
            mlx_rss.last().expect("RSS history empty")
                < mlx_rss.first().expect("RSS history empty"),
            "MLX RSS did not decrease (pruning {}): {:?}",
            pruning,
            mlx_rss
        );

        let agree = cpu_assign
            .iter()
            .zip(mlx_assign.iter())
            .filter(|(a, b)| a == b)
            .count();
        assert!(
            agree * 100 >= n * 95,
            "only {} / {} assignments agree (pruning {})",
            agree,
            n,
            pruning
        );

        let cpu_final = cpu_rss[cpu_rss.len() - 1];
        let mlx_final = mlx_rss[mlx_rss.len() - 1];
        let rel = ((cpu_final - mlx_final) / cpu_final).abs();
        assert!(
            rel < 1e-3,
            "RSS differs (pruning {}): {} vs {} (rel {:.2e})",
            pruning,
            cpu_final,
            mlx_final,
            rel
        );

        // Every cell lands in exactly one metacell.
        let total: usize = mlx_metacells.iter().map(|m| m.len()).sum();
        assert_eq!(
            total, n,
            "metacell grouping lost cells (pruning {})",
            pruning
        );
    }
}
