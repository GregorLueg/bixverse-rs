//! Micro-benchmark for the SEACells B-gradient argmin kernel.
//!
//! Rebuilds a `K²B` of realistic shape and density, times `launch_fw_argmin_b`
//! against the per-iteration CPU cost recorded in [Shape], then runs the full fit
//! both ways. `BIXVERSE_BENCH_BIG=1` adds shapes up to 500k cells.
//!
//! Timings are best-of-N with the worst also reported: a single-shot number is a
//! first-call number, since shader compilation and buffer-pool first touch land
//! entirely on run one.
//!
//! ```bash
//! cargo bench --features gpu,single-cell --bench seacells_gpu_bench
//! ```

use std::time::{Duration, Instant};

use ann_search_rs::gpu::tensor::GpuTensor;
use cubecl::future;
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use faer::Mat;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use bixverse_rs::gpu::linalg::sparse_gpu::GpuCompressedSparseData;
use bixverse_rs::gpu::sc_gpu::kernels::seacells_kernels::{B_ARGMIN_BLOCKS, launch_fw_argmin_b};
use bixverse_rs::gpu::sc_gpu::seacells_gpu::seacells_fit_gpu;
use bixverse_rs::prelude::CompressedSparseFormat;
use bixverse_rs::single_cell::mc_generation::seacells::{SEACells, SEACellsParams};
use bixverse_rs::single_cell::sc_processing::knn::{KnnParams, generate_knn_with_dist};

////////////
// Consts //
////////////

/// Repeats per shape. The first is dominated by shader compilation and first
/// touch, so more than a couple is needed before the steady state is visible.
const REPEATS: usize = 5;

//////////////////
// Bench shapes //
//////////////////

/// One measured configuration, matching a row of the CPU attribution.
#[derive(Clone, Copy, Debug)]
struct Shape {
    /// Number of cells
    n: usize,
    /// Number of archetypes
    k: usize,
    /// Fraction of `n * k` that is non-zero in `K²B`
    density: f64,
    /// Measured CPU cost per Frank-Wolfe iteration, milliseconds, where a
    /// baseline exists
    cpu_ms: Option<f64>,
}

/// Shapes taken straight from the post-rewrite CPU attribution.
///
/// ### Returns
///
/// The shapes to measure.
fn shapes() -> Vec<Shape> {
    let mut out = vec![
        Shape {
            n: 20_000,
            k: 266,
            density: 0.0341,
            cpu_ms: Some(1363.0 / 150.0),
        },
        Shape {
            n: 50_000,
            k: 666,
            density: 0.0308,
            cpu_ms: Some(12280.0 / 150.0),
        },
        Shape {
            n: 50_000,
            k: 200,
            density: 0.0405,
            cpu_ms: Some(5500.0 / 150.0),
        },
    ];

    // Large shapes probe where the kernel degrades. `t1` is dense k x k, so it
    // leaves cache somewhere around k = 3500 on a 48 MB SLC, and the inner loop
    // reads a full t1 row per non-zero of K2B. Densities here are extrapolated
    // from the measured trend (3.4% at 20k, 3.1% at 50k, 2.3% at 200k), so the
    // absolute times scale with whatever the real density turns out to be.
    if std::env::var("BIXVERSE_BENCH_BIG").is_ok() {
        out.extend([
            Shape {
                n: 200_000,
                k: 1_000,
                density: 0.03,
                cpu_ms: None,
            },
            Shape {
                n: 500_000,
                k: 1_000,
                density: 0.03,
                cpu_ms: None,
            },
            Shape {
                n: 500_000,
                k: 2_000,
                density: 0.02,
                cpu_ms: None,
            },
            Shape {
                n: 500_000,
                k: 4_000,
                density: 0.01,
                cpu_ms: None,
            },
            Shape {
                n: 500_000,
                k: 6_666,
                density: 0.006,
                cpu_ms: None,
            },
        ]);
    }
    out
}

/////////////
// Fixture //
/////////////

/// Build CSR parts for an `n × k` matrix at the requested density.
///
/// Non-zeros are spread evenly across each row with a per-row phase offset, so
/// the column indices are sorted and the row lengths are realistic without
/// needing an RNG.
///
/// ### Params
///
/// * `n` - Rows
/// * `k` - Columns
/// * `density` - Fraction of entries that are non-zero
/// * `seed` - Shifts the phase and the values
///
/// ### Returns
///
/// `(values, indices, indptr)`.
fn make_csr(n: usize, k: usize, density: f64, seed: usize) -> (Vec<f32>, Vec<u32>, Vec<u32>) {
    let per_row = ((k as f64 * density).round() as usize).max(1);
    let stride = (k / per_row).max(1);

    let mut values = Vec::with_capacity(n * per_row);
    let mut indices = Vec::with_capacity(n * per_row);
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0u32);

    for row in 0..n {
        let phase = (row * 7 + seed) % stride;
        let mut col = phase;
        while col < k {
            indices.push(col as u32);
            values.push((((row * 31 + col * 17 + seed) % 19) as f32 - 9.0) * 0.05);
            col += stride;
        }
        indptr.push(values.len() as u32);
    }

    (values, indices, indptr)
}

//////////
// Main //
//////////

/// Time the kernel pair on one shape.
///
/// ### Params
///
/// * `shape` - The configuration
/// * `device` - Device to run on
fn run_shape(shape: Shape, device: &WgpuDevice) {
    let client = WgpuRuntime::client(device);
    let Shape { n, k, density, .. } = shape;

    let upload = |parts: (Vec<f32>, Vec<u32>, Vec<u32>)| {
        GpuCompressedSparseData::<WgpuRuntime, f32>::from_parts(
            &parts.0,
            &parts.1,
            &parts.2,
            CompressedSparseFormat::Csr,
            (n, k),
            &client,
        )
    };

    let k2b_parts = make_csr(n, k, density, 1);
    let nnz = k2b_parts.0.len();
    let k2b = upload(k2b_parts);
    let t2 = upload(make_csr(n, k, density * 0.6, 5));
    // B carries roughly `max_fw_iters` atoms per archetype column, so it is far
    // sparser than K²B.
    let b_mat = upload(make_csr(n, k, 25.0 / n as f64, 9));

    let t1_host: Vec<f32> = (0..k * k)
        .map(|i| (((i * 13) % 17) as f32 - 8.0) * 0.02)
        .collect();
    let t1 = GpuTensor::<WgpuRuntime, f32>::from_slice(&t1_host, vec![k * k], &client);

    let blocks = B_ARGMIN_BLOCKS.min(n.max(1) as u32) as usize;
    let part_val = GpuTensor::<WgpuRuntime, f32>::empty(vec![blocks * k], &client);
    let part_idx = GpuTensor::<WgpuRuntime, u32>::empty(vec![blocks * k], &client);
    let gap_partial = GpuTensor::<WgpuRuntime, f32>::empty(vec![blocks], &client);
    let out_val = GpuTensor::<WgpuRuntime, f32>::empty(vec![k], &client);
    let out_idx = GpuTensor::<WgpuRuntime, u32>::empty(vec![k], &client);

    let mut timings: Vec<Duration> = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let start = Instant::now();
        launch_fw_argmin_b(
            &k2b,
            &t1,
            &t2,
            &b_mat,
            &part_val,
            &part_idx,
            &gap_partial,
            &out_val,
            &out_idx,
            n,
            k,
            &client,
        )
        .expect("launch failed");
        future::block_on(client.sync()).expect("device sync failed");
        timings.push(start.elapsed());
    }

    timings.sort();
    let best = timings[0].as_secs_f64() * 1000.0;
    let worst = timings[REPEATS - 1].as_secs_f64() * 1000.0;

    // A launch that busts a device limit fails silently and returns zeros, so
    // the output has to be checked before any timing is believed.
    let vals = out_val.clone().read(&client).expect("read failed");
    let idx = out_idx.clone().read(&client).expect("read failed");
    let nonzero = vals.iter().filter(|v| **v != 0.0 && v.is_finite()).count();
    assert!(
        nonzero > k / 2,
        "kernel almost certainly did no work: only {} / {} finite non-zero minima",
        nonzero,
        k
    );
    assert!(
        idx.iter().any(|i| *i != 0),
        "every argmin came back as row 0, which is not plausible"
    );

    let flops = nnz as f64 * k as f64;
    let vram = k2b.vram_bytes()
        + t2.vram_bytes()
        + b_mat.vram_bytes()
        + t1.vram_bytes()
        + part_val.vram_bytes()
        + part_idx.vram_bytes();
    let speedup = match shape.cpu_ms {
        Some(cpu) => format!("{:>6.1}x", cpu / best),
        None => "     -".to_string(),
    };

    println!(
        "n = {:>7}  k = {:>5}  nnz(K2B) = {:>10}  |  GPU best {:>8.2} ms  worst {:>8.2} ms  \
         |  speedup {}  |  {:>6.1} GFLOP/s  |  {:>7.1} MB VRAM  |  t1 {:>6.1} MB",
        n,
        k,
        nnz,
        best,
        worst,
        speedup,
        flops / (best / 1000.0) / 1e9,
        vram as f64 / (1024.0 * 1024.0),
        (k * k * 4) as f64 / (1024.0 * 1024.0),
    );
}

//////////////
// End-to-end   //
//////////////////

/// Clustered synthetic embedding, matching `benches/seacells_bench.rs`.
///
/// ### Params
///
/// * `n` - Number of cells
/// * `dim` - Embedding width
/// * `seed` - Random seed
///
/// ### Returns
///
/// An `n × dim` matrix.
fn make_embedding(n: usize, dim: usize, seed: u64) -> Mat<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centre_dist = Normal::new(0.0f32, 1.0).expect("valid normal");
    let noise_dist = Normal::new(0.0f32, 0.55).expect("valid normal");

    let centres: Vec<Vec<f32>> = (0..40)
        .map(|_| (0..dim).map(|_| centre_dist.sample(&mut rng)).collect())
        .collect();
    let assignments: Vec<usize> = (0..n).map(|_| rng.random_range(0..40)).collect();

    Mat::from_fn(n, dim, |i, j| {
        centres[assignments[i]][j] + noise_dist.sample(&mut rng)
    })
}

/// Parameters for the end-to-end runs.
///
/// ### Params
///
/// * `n_sea_cells` - Number of SEACells
/// * `knn` - Neighbours in the kNN graph
///
/// ### Returns
///
/// The populated [SEACellsParams].
fn e2e_params(n_sea_cells: usize, knn: usize) -> SEACellsParams {
    let mut knn_params = KnnParams::new();
    knn_params.k = knn;
    knn_params.ann_dist = "euclidean".to_string();

    SEACellsParams {
        n_sea_cells,
        max_fw_iters: 50,
        convergence_epsilon: 1e-3,
        max_iter: 3,
        min_iter: 3,
        greedy_threshold: 0,
        graph_building: "union".to_string(),
        pruning: true,
        pruning_threshold: 1e-7,
        n_landmarks: None,
        knn_params,
    }
}

/// Run the full fit on both paths and compare time and result.
///
/// ### Params
///
/// * `n` - Number of cells
/// * `k` - Number of SEACells
/// * `knn` - Neighbours in the kNN graph
/// * `device` - Device for the GPU arm
fn run_end_to_end(n: usize, k: usize, knn: usize, device: &WgpuDevice) {
    let params = e2e_params(k, knn);
    let embedding = make_embedding(n, 30, 7);

    let (knn_indices, knn_distances) = generate_knn_with_dist(
        embedding.as_ref(),
        &params.knn_params,
        true,
        false,
        42,
        false,
    )
    .expect("kNN failed");
    let knn_distances = knn_distances.expect("distances requested");

    let cpu_start = Instant::now();
    let mut cpu_model = SEACells::new(n, &params);
    cpu_model.construct_kernel_mat(embedding.as_ref(), &knn_indices, &knn_distances, 0);
    cpu_model
        .initialise_archetypes(&knn_indices, &knn_distances, 0, true, 42)
        .expect("archetype init failed");
    cpu_model.fit(42, 0).expect("CPU fit failed");
    let cpu_time = cpu_start.elapsed();
    let cpu_assign = cpu_model
        .get_hard_assignments()
        .expect("assignments failed");
    let cpu_rss = cpu_model.get_rss_history().to_vec();

    let gpu_start = Instant::now();
    let (gpu_assign, _, _, gpu_rss) = seacells_fit_gpu::<WgpuRuntime>(
        embedding.as_ref(),
        &knn_indices,
        &knn_distances,
        true,
        &params,
        42,
        device.clone(),
        0,
    )
    .expect("GPU fit failed");
    let gpu_time = gpu_start.elapsed();

    let agree = cpu_assign
        .iter()
        .zip(gpu_assign.iter())
        .filter(|(a, b)| a == b)
        .count();

    println!(
        "n = {:>6}  k = {:>4}  |  CPU {:>7.2}s  GPU {:>7.2}s  |  speedup {:>5.2}x  \
         |  assignments agree {:.1}%  |  RSS {:.4} vs {:.4}",
        n,
        k,
        cpu_time.as_secs_f64(),
        gpu_time.as_secs_f64(),
        cpu_time.as_secs_f64() / gpu_time.as_secs_f64(),
        100.0 * agree as f64 / n as f64,
        cpu_rss[cpu_rss.len() - 1],
        gpu_rss[gpu_rss.len() - 1],
    );

    assert!(
        gpu_rss[gpu_rss.len() - 1] < gpu_rss[0],
        "GPU RSS did not decrease: {:?}",
        gpu_rss
    );
}

fn main() {
    let device = WgpuDevice::DefaultDevice;
    println!(
        "\nSEACells B-gradient argmin: GPU kernel vs measured CPU baseline\n\
         (CPU numbers from benches/seacells_bench.rs, pruning 1e-7)\n"
    );
    for shape in shapes() {
        run_shape(shape, &device);
    }

    println!("\nEnd to end, full fit, pruning 1e-7, 3 outer iterations\n");
    for (n, k, knn) in [(20_000usize, 266usize, 15usize), (50_000, 666, 15)] {
        run_end_to_end(n, k, knn, &device);
    }

    // GPU only at a larger shape, where the CPU arm would take tens of minutes.
    if std::env::var("BIXVERSE_BENCH_BIG").is_ok() {
        println!("\nGPU only at scale\n");
        {
            let (n, k, knn) = (200_000usize, 2_666usize, 15usize);
            let params = e2e_params(k, knn);
            let embedding = make_embedding(n, 30, 7);
            let (knn_indices, knn_distances) = generate_knn_with_dist(
                embedding.as_ref(),
                &params.knn_params,
                true,
                false,
                42,
                false,
            )
            .expect("kNN failed");
            let knn_distances = knn_distances.expect("distances requested");

            let start = Instant::now();
            let (_, _, _, rss) = seacells_fit_gpu::<WgpuRuntime>(
                embedding.as_ref(),
                &knn_indices,
                &knn_distances,
                true,
                &params,
                42,
                device.clone(),
                0,
            )
            .expect("GPU fit failed");
            println!(
                "n = {} k = {} GPU fit in {:.2}s, RSS {:.4} -> {:.4}",
                n,
                k,
                start.elapsed().as_secs_f64(),
                rss[0],
                rss[rss.len() - 1]
            );
        }
    }
    println!();
}
