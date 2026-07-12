//! CPU vs GPU wall-clock for SCENIC multi-tree regression.
//!
//! Two bench matrices measure different things:
//!
//! - **kernel**: single 64-target batch, low-level `fit_multi_trees_sparse`
//!   / `fit_multi_trees_gpu`. Per-batch kernel throughput. Regression guard
//!   for cross-batch amortisation work: this number should not get worse.
//! - **end_to_end**: multi-batch workload via `run_scenic_grn_in_memory`
//!   / `run_scenic_grn_in_memory_gpu`, ~4000 targets in ~63 batches. Exposes
//!   CPU's rayon batch fan-out (rayon `into_par_iter` over batches, one
//!   `fit_multi_trees_sparse` per task) versus GPU's strictly sequential
//!   `for batch in ...` loop with per-batch feature re-upload, `WaveState`
//!   alloc and blocking readback.
//!
//! Cell count is fixed at 10k for a fast baseline / regression run. Bump the
//! constants when you want a full sweep.
//!
//! Wave VRAM budget is bumped to 12 GiB so `wave_size = 1` fits comfortably
//! at 10k cells; at the default 4 GiB `pick_wave_size` errors past 50k.
//!
//! Run with:
//! ```
//! cargo bench --features gpu,single-cell --bench gpu_scenic_bench
//! ```

#![cfg(all(feature = "gpu", feature = "single-cell"))]
#![allow(clippy::field_reassign_with_default, clippy::needless_range_loop)]

use std::time::Instant;

use bixverse_rs::gpu::sc_gpu::scenic_gpu::{
    ScenicGpuParams, fit_multi_trees_gpu, run_scenic_grn_in_memory_gpu,
};
use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::mc_analysis::scenic_metacells::run_scenic_grn_in_memory;
use bixverse_rs::single_cell::sc_analysis::scenic::{
    ExtraTreesConfig, RandomForestConfig, RegressionLearner, ScenicParams, TreeRegressorConfig,
    fit_multi_trees_sparse,
};
use bixverse_rs::single_cell::sc_utils::utils_tree::QuantisedStore;

use cubecl::Runtime;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use rand::prelude::*;
use rand::rngs::SmallRng;

////////////
// Shapes //
////////////

/// Cell counts for both matrices. Kept small for a fast baseline run; extend
/// this array when doing a full sweep.
const CELL_COUNTS: &[usize] = &[10_000];

/// Kernel matrix: single 64-target batch, low-level fitter.
const N_FEATURES_KERNEL: usize = 1_000;
const N_TARGETS_KERNEL: usize = 64;

/// End-to-end matrix: realistic multi-batch shape. 4000 targets at batch
/// size 64 = 63 batches, enough to expose the rayon fan-out gap.
const N_TFS_E2E: usize = 1_000;
const N_TARGETS_E2E: usize = 4_000;

const N_TREES: usize = 250;
const MAX_DEPTH: usize = 10;
const MIN_SAMPLES_LEAF: usize = 50;
const N_FEATURES_SPLIT: usize = 0;
const N_INFORMATIVE: usize = 10;
const SPARSITY: f32 = 0.5;
const SEED: usize = 20260708;

/// Ceiling per bench iteration (median-of-3). Any shape whose warmup alone
/// exceeds this is skipped for that variant.
const SKIP_ABOVE_SECS: f32 = 300.0;

/// 12 GiB. Lets `pick_wave_size` land wave=1 even at 100k cells with 64
/// targets / sqrt(1000) features per split. The default 4 GiB errors past
/// 50k cells on this shape.
const BENCH_WAVE_BUDGET: usize = 12 * 1024 * 1024 * 1024;

////////////////
// Data build //
////////////////

/// Kernel-matrix quantised feature store: raw u8 bins.
fn make_features_kernel(n_samples: usize, seed: u64) -> QuantisedStore {
    let mut rng = SmallRng::seed_from_u64(seed);
    let data: Vec<u8> = (0..n_samples * N_FEATURES_KERNEL)
        .map(|_| rng.random())
        .collect();
    QuantisedStore::from_raw(data, n_samples, N_FEATURES_KERNEL)
}

/// Kernel-matrix synthetic targets: 10 informative features, boosted weights.
fn make_targets_kernel(
    x: &QuantisedStore,
    n_samples: usize,
    seed: u64,
) -> Vec<SparseAxis<u32, f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut weight_rng = SmallRng::seed_from_u64(0xF00D_BABE);
    let mut weights = vec![vec![0.0f32; N_INFORMATIVE]; N_TARGETS_KERNEL];
    for w_row in weights.iter_mut() {
        for w in w_row.iter_mut() {
            *w = weight_rng.random::<f32>() + 1.0;
        }
    }

    let feats: Vec<&[u8]> = (0..N_INFORMATIVE).map(|f| x.get_col(f)).collect();

    let mut cols_indices: Vec<Vec<usize>> = vec![Vec::new(); N_TARGETS_KERNEL];
    let mut cols_values: Vec<Vec<f32>> = vec![Vec::new(); N_TARGETS_KERNEL];
    for c in 0..n_samples {
        for t in 0..N_TARGETS_KERNEL {
            if rng.random::<f32>() < SPARSITY {
                let mut signal = 0.0f32;
                for f in 0..N_INFORMATIVE {
                    signal += weights[t][f] * (feats[f][c] as f32 / 255.0);
                }
                let noise: f32 = rng.random::<f32>() * 0.05;
                cols_indices[t].push(c);
                cols_values[t].push(signal + noise + 0.01);
            }
        }
    }
    cols_indices
        .into_iter()
        .zip(cols_values)
        .map(|(idx, vs)| SparseAxis::<u32, f32>::new_csc(idx, Vec::new(), Some(vs), n_samples))
        .collect()
}

/// End-to-end matrix: synthetic cells x genes CSC. All columns are treated
/// as targets by the top-level drivers; `tf_indices` picks the first
/// `N_TFS_E2E` as predictors.
fn make_csc_e2e(n_cells: usize, seed: u64) -> CompressedSparseData2<u16, f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n_cols = N_TARGETS_E2E;

    let mut data: Vec<u16> = Vec::new();
    let mut data_2: Vec<f32> = Vec::new();
    let mut indices: Vec<usize> = Vec::new();
    let mut indptr: Vec<usize> = vec![0];

    // Cache first N_INFORMATIVE (which sit inside the TF range) so downstream
    // targets can carry real signal instead of pure noise.
    let mut tf_cache: Vec<Vec<f32>> = vec![vec![0.0f32; n_cells]; N_INFORMATIVE];

    for j in 0..n_cols {
        for c in 0..n_cells {
            if rng.random::<f32>() < SPARSITY {
                // Informative TFs get random baseline; other columns get
                // random baseline + summed signal from the cached TFs.
                let v: f32 = if j < N_INFORMATIVE {
                    let raw: f32 = rng.random_range(0.0..5.0);
                    tf_cache[j][c] = raw;
                    raw
                } else {
                    let mut signal = 0.0f32;
                    for f in 0..N_INFORMATIVE {
                        signal += 0.3 * tf_cache[f][c];
                    }
                    let noise: f32 = rng.random::<f32>();
                    (signal + noise).max(0.5)
                };
                if v > 1.0 {
                    indices.push(c);
                    data.push(v as u16);
                    data_2.push(v);
                }
            }
        }
        indptr.push(data.len());
    }

    CompressedSparseData2 {
        data,
        indices: indices.into_iter().map(|i| i as u32).collect(),
        indptr: indptr.into_iter().map(|i| i as u32).collect(),
        cs_type: CompressedSparseFormat::Csc,
        data_2: Some(data_2),
        shape: (n_cells, n_cols),
    }
}

/////////////
// Configs //
/////////////

fn et_config() -> ExtraTreesConfig {
    let mut c = ExtraTreesConfig::default();
    c.n_trees = N_TREES;
    c.max_depth = Some(MAX_DEPTH);
    c.min_samples_leaf = MIN_SAMPLES_LEAF;
    c.n_features_split = N_FEATURES_SPLIT;
    c
}

fn rf_config() -> RandomForestConfig {
    let mut c = RandomForestConfig::default();
    c.n_trees = N_TREES;
    c.max_depth = Some(MAX_DEPTH);
    c.min_samples_leaf = MIN_SAMPLES_LEAF;
    c.n_features_split = N_FEATURES_SPLIT;
    c
}

/// Build a full [`ScenicParams`] suitable for the end-to-end matrix.
///
/// - `random` gene batching to avoid the PCA overhead of `correlated`;
/// - `min_counts` / `min_cells` zeroed so nothing gets filtered out.
fn scenic_params_e2e(learner: RegressionLearner) -> ScenicParams {
    ScenicParams {
        min_counts: 0,
        min_cells: 0.0,
        regression_learner: learner,
        gene_batch_strategy: "random".to_string(),
        gene_batch_size: None,
        n_pcs: 50,
        n_subsample: 100_000,
    }
}

fn try_device() -> Option<WgpuDevice> {
    let device = WgpuDevice::DefaultDevice;
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        WgpuRuntime::client(&device);
    }))
    .ok()
    .map(|_| device)
}

////////////
// Timing //
////////////

fn label(cells: usize) -> String {
    format!("{}k", cells / 1000)
}

/// One warmup + 3 measured iterations, median in seconds. Returns `None` if
/// the warmup alone crosses [`SKIP_ABOVE_SECS`].
fn median_of_3<F>(mut f: F) -> Option<f32>
where
    F: FnMut(),
{
    let t0 = Instant::now();
    f();
    let warm = t0.elapsed().as_secs_f32();
    if warm > SKIP_ABOVE_SECS {
        return None;
    }

    let mut times = Vec::with_capacity(3);
    for _ in 0..3 {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_secs_f32());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(times[1])
}

////////////////////
// Kernel matrix  //
////////////////////

fn run_kernel_cpu(
    n_samples: usize,
    x: &QuantisedStore,
    axes: &[SparseAxis<u32, f32>],
    config: &dyn TreeRegressorConfig,
    id: &str,
) {
    println!("  {id}: running (CPU)...");
    match median_of_3(|| {
        fit_multi_trees_sparse(axes, x, n_samples, config, SEED).expect("CPU fit failed");
    }) {
        Some(t) => println!("  {id}: {t:.2}s"),
        None => println!("  {id}: SKIPPED (warmup > {SKIP_ABOVE_SECS:.0}s)"),
    }
    let _ = n_samples;
}

fn run_kernel_gpu(
    n_samples: usize,
    x: &QuantisedStore,
    axes: &[SparseAxis<u32, f32>],
    config: &dyn TreeRegressorConfig,
    device: &WgpuDevice,
    id: &str,
) {
    let params = ScenicGpuParams {
        wave_byte_budget: BENCH_WAVE_BUDGET,
    };
    println!("  {id}: running (GPU)...");
    match median_of_3(|| {
        fit_multi_trees_gpu::<WgpuRuntime>(
            axes,
            x,
            n_samples,
            config,
            SEED,
            device.clone(),
            &params,
        )
        .expect("GPU fit failed");
    }) {
        Some(t) => println!("  {id}: {t:.2}s"),
        None => println!("  {id}: SKIPPED (warmup > {SKIP_ABOVE_SECS:.0}s)"),
    }
}

fn kernel_matrix(device: &WgpuDevice) {
    println!(
        "\n=== kernel matrix: {} TFs x {} targets (1 batch) x {} trees ===",
        N_FEATURES_KERNEL, N_TARGETS_KERNEL, N_TREES,
    );
    let et = et_config();
    let rf = rf_config();

    for &n in CELL_COUNTS {
        let l = label(n);
        println!("--- kernel shape {n} cells ---");

        let x = make_features_kernel(n, SEED as u64 + n as u64);
        let axes = make_targets_kernel(&x, n, SEED as u64 + n as u64 + 1);

        run_kernel_cpu(n, &x, &axes, &et, &format!("et_cpu_{l}"));
        run_kernel_gpu(n, &x, &axes, &et, device, &format!("et_gpu_{l}"));
        run_kernel_cpu(n, &x, &axes, &rf, &format!("rf_cpu_{l}"));
        run_kernel_gpu(n, &x, &axes, &rf, device, &format!("rf_gpu_{l}"));
        println!();
    }
}

////////////////////////
// End-to-end matrix  //
////////////////////////

/// Single measured iteration with no warmup, so the bench can complete on
/// GPU without hitting the kernel-matrix skip threshold. E2E numbers land
/// in the many-hundreds-of-seconds range and one honest sample is enough
/// to see whether GPU is winning or losing.
fn single_shot<F: FnOnce()>(f: F) -> f32 {
    let t0 = Instant::now();
    f();
    t0.elapsed().as_secs_f32()
}

fn run_e2e_cpu(csc: &CompressedSparseData2<u16, f32>, params: &ScenicParams, id: &str) {
    println!("  {id}: running (CPU)...");
    let t = single_shot(|| {
        let tf_indices: Vec<usize> = (0..N_TFS_E2E).collect();
        run_scenic_grn_in_memory(csc, &tf_indices, params, SEED, 0).expect("CPU e2e fit failed");
    });
    println!("  {id}: {t:.2}s");
}

fn run_e2e_gpu(
    csc: &CompressedSparseData2<u16, f32>,
    params: &ScenicParams,
    device: &WgpuDevice,
    id: &str,
) {
    let gpu_params = ScenicGpuParams {
        wave_byte_budget: BENCH_WAVE_BUDGET,
    };
    println!("  {id}: running (GPU)...");
    let t = single_shot(|| {
        let tf_indices: Vec<usize> = (0..N_TFS_E2E).collect();
        run_scenic_grn_in_memory_gpu::<WgpuRuntime, _>(
            csc,
            &tf_indices,
            params,
            &gpu_params,
            SEED,
            device.clone(),
            0,
        )
        .expect("GPU e2e fit failed");
    });
    println!("  {id}: {t:.2}s");
}

fn end_to_end_matrix(device: &WgpuDevice) {
    println!(
        "\n=== end_to_end matrix: {} TFs x {} targets (~{} batches) x {} trees ===",
        N_TFS_E2E,
        N_TARGETS_E2E,
        N_TARGETS_E2E.div_ceil(64),
        N_TREES,
    );

    let et_params = scenic_params_e2e(RegressionLearner::ExtraTrees(et_config()));
    let rf_params = scenic_params_e2e(RegressionLearner::RandomForest(rf_config()));

    for &n in CELL_COUNTS {
        let l = label(n);
        println!("--- e2e shape {n} cells ---");

        let csc = make_csc_e2e(n, SEED as u64 + n as u64 + 2);

        run_e2e_cpu(&csc, &et_params, &format!("et_e2e_cpu_{l}"));
        run_e2e_gpu(&csc, &et_params, device, &format!("et_e2e_gpu_{l}"));
        run_e2e_cpu(&csc, &rf_params, &format!("rf_e2e_cpu_{l}"));
        run_e2e_gpu(&csc, &rf_params, device, &format!("rf_e2e_gpu_{l}"));
        println!();
    }
}

fn main() {
    println!("gpu_scenic_bench: median-of-3 wall clock");
    println!("  skip threshold: warmup > {SKIP_ABOVE_SECS:.0}s");

    let device = match try_device() {
        Some(d) => d,
        None => {
            eprintln!("no wgpu device available -- aborting");
            std::process::exit(1);
        }
    };

    kernel_matrix(&device);
    end_to_end_matrix(&device);
}
