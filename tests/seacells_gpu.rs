//! End-to-end parity between the CPU and GPU SEACells paths.
//!
//! Only the two Frank-Wolfe inner solves differ between the two paths, the
//! B-gradient argmin and the A-column update, so the trajectories should track
//! each other closely. Near-ties in either argmin can still resolve differently
//! between a scattered CPU accumulation and a register-blocked GPU one, so the
//! assertions are on agreement rate and the objective rather than on exact
//! equality.
//!
//! The single test here is the most expensive in the crate: n = 3000 over two
//! pruning arms, each running a full CPU SEACells fit and a full GPU fit. The
//! whole file is therefore behind `large_scale_diagnostics`.

#![cfg(all(
    feature = "single-cell",
    feature = "gpu",
    feature = "large_scale_diagnostics"
))]

use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use faer::Mat;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use bixverse_rs::gpu::sc_gpu::seacells_gpu::seacells_fit_gpu;
use bixverse_rs::single_cell::mc_generation::seacells::{SEACells, SEACellsParams};
use bixverse_rs::single_cell::sc_processing::knn::{KnnParams, generate_knn_with_dist};

/// Skip rather than fail where no GPU is available.
///
/// ### Returns
///
/// The default device, or `None` if a client cannot be created.
fn try_device() -> Option<WgpuDevice> {
    let device = WgpuDevice::DefaultDevice;
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        WgpuRuntime::client(&device);
    }))
    .ok()
    .map(|_| device)
}

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
/// An `n × dim` matrix.
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
    }
}

/// The GPU path must reach the same solution as the CPU path, with pruning both
/// off and on.
#[test]
fn test_seacells_gpu_matches_cpu() {
    let Some(device) = try_device() else { return };

    let n = 3000usize;
    let k = 40usize;
    let embedding = make_embedding(n, 20, 12, 7);

    for pruning in [false, true] {
        let p = params(k, pruning);

        let (knn_indices, knn_distances) =
            generate_knn_with_dist(embedding.as_ref(), &p.knn_params, true, false, 42, false)
                .expect("kNN failed");
        let knn_distances = knn_distances.expect("distances requested");

        let mut cpu_model = SEACells::new(n, &p);
        cpu_model.construct_kernel_mat(embedding.as_ref(), &knn_indices, &knn_distances, 0);
        cpu_model
            .initialise_archetypes(&knn_indices, &knn_distances, 0, true, 42)
            .expect("archetype init failed");
        cpu_model.fit(42, 0).expect("CPU fit failed");
        let cpu_assign = cpu_model
            .get_hard_assignments()
            .expect("assignments failed");
        let cpu_rss = cpu_model.get_rss_history().to_vec();

        let (gpu_assign, gpu_metacells, _, gpu_rss) = seacells_fit_gpu::<WgpuRuntime>(
            embedding.as_ref(),
            &knn_indices,
            &knn_distances,
            true,
            &p,
            42,
            device.clone(),
            0,
        )
        .expect("GPU fit failed");

        // A launch that busts a device limit fails silently and writes nothing,
        // so check the run did work before comparing anything.
        assert_eq!(gpu_assign.len(), n);
        assert!(
            gpu_rss.last().expect("RSS history empty")
                < gpu_rss.first().expect("RSS history empty"),
            "GPU RSS did not decrease (pruning {}): {:?}",
            pruning,
            gpu_rss
        );

        let agree = cpu_assign
            .iter()
            .zip(gpu_assign.iter())
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
        let gpu_final = gpu_rss[gpu_rss.len() - 1];
        let rel = ((cpu_final - gpu_final) / cpu_final).abs();
        assert!(
            rel < 1e-3,
            "RSS differs (pruning {}): {} vs {} (rel {:.2e})",
            pruning,
            cpu_final,
            gpu_final,
            rel
        );

        // Every cell lands in exactly one metacell.
        let total: usize = gpu_metacells.iter().map(|m| m.len()).sum();
        assert_eq!(
            total, n,
            "metacell grouping lost cells (pruning {})",
            pruning
        );
    }
}
