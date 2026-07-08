//! Phase 4a: CPU vs GPU wall-clock for SCENIC multi-tree regression.
//!
//! Bench matrix: cell counts {10k, 25k, 50k, 75k, 100k} x learner {ET, RF} x
//! backend {CPU, GPU}, with 1_000 TFs, 64 targets (one batch), 250 trees,
//! max_depth=10, min_samples_leaf=50, n_features_split=0 (auto = sqrt(n_feats)).
//!
//! Synthetic data mirrors the Phase 2 test harness: 10 informative features
//! carrying signal for every target, boosted weights so ExtraTrees has real
//! signal to learn rather than pure noise.
//!
//! Wave VRAM budget is bumped to 12 GiB so wave=1 fits even at 100k cells;
//! at the default 4 GiB `pick_wave_size` errors past 50k cells.
//!
//! Not a criterion harness -- criterion's minimum 10 samples pushes CPU RF at
//! 50k+ into hours. Plain `Instant` timing with 1 warmup + 3 measured iters,
//! reports median wall-clock per shape.
//!
//! Run with:
//! ```
//! cargo bench --features gpu,single-cell --bench gpu_scenic_bench
//! ```

#![cfg(all(feature = "gpu", feature = "single-cell"))]
#![allow(clippy::field_reassign_with_default, clippy::needless_range_loop)]

use std::time::Instant;

use bixverse_rs::gpu::sc_gpu::scenic_gpu::fit_multi_trees_gpu;
use bixverse_rs::gpu::sc_gpu::scenic_gpu_params::ScenicGpuParams;
use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_analysis::scenic::{
    ExtraTreesConfig, RandomForestConfig, TreeRegressorConfig, fit_multi_trees_sparse,
};
use bixverse_rs::single_cell::sc_utils::utils_tree::QuantisedStore;

use cubecl::Runtime;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use rand::prelude::*;
use rand::rngs::SmallRng;

////////////
// Shapes //
////////////

const CELL_COUNTS: &[usize] = &[10_000, 25_000, 50_000, 75_000, 100_000];
const N_FEATURES: usize = 1_000;
const N_TARGETS: usize = 64;
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

fn make_features(n_samples: usize, seed: u64) -> QuantisedStore {
    let mut rng = SmallRng::seed_from_u64(seed);
    let data: Vec<u8> = (0..n_samples * N_FEATURES).map(|_| rng.random()).collect();
    QuantisedStore::from_raw(data, n_samples, N_FEATURES)
}

fn make_targets(x: &QuantisedStore, n_samples: usize, seed: u64) -> Vec<SparseAxis<u32, f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);

    // fixed weights so target structure is stable across seeds
    let mut weight_rng = SmallRng::seed_from_u64(0xF00D_BABE);
    let mut weights = vec![vec![0.0f32; N_INFORMATIVE]; N_TARGETS];
    for w_row in weights.iter_mut() {
        for w in w_row.iter_mut() {
            *w = weight_rng.random::<f32>() + 1.0;
        }
    }

    let feats: Vec<&[u8]> = (0..N_INFORMATIVE).map(|f| x.get_col(f)).collect();

    let mut cols_indices: Vec<Vec<usize>> = vec![Vec::new(); N_TARGETS];
    let mut cols_values: Vec<Vec<f32>> = vec![Vec::new(); N_TARGETS];
    for c in 0..n_samples {
        for t in 0..N_TARGETS {
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

fn try_device() -> Option<WgpuDevice> {
    let device = WgpuDevice::DefaultDevice;
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        WgpuRuntime::client(&device);
    }))
    .ok()
    .map(|_| device)
}

/////////////
// Timing  //
/////////////

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

fn run_cpu(
    n_samples: usize,
    x: &QuantisedStore,
    axes: &[SparseAxis<u32, f32>],
    config: &dyn TreeRegressorConfig,
    id: &str,
) {
    println!("  {id}: running (CPU)...");
    match median_of_3(|| {
        fit_multi_trees_sparse(axes, x, n_samples, config, SEED)
            .expect("CPU fit failed");
    }) {
        Some(t) => println!("  {id}: {t:.2}s"),
        None => println!("  {id}: SKIPPED (warmup > {SKIP_ABOVE_SECS:.0}s)"),
    }
}

fn run_gpu(
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

fn main() {
    println!("gpu_scenic_bench (phase 4a): median-of-3 wall clock");
    println!(
        "  shape: {} TFs, {} targets, {} trees, max_depth={}, min_samples_leaf={}",
        N_FEATURES, N_TARGETS, N_TREES, MAX_DEPTH, MIN_SAMPLES_LEAF
    );
    println!("  skip threshold: warmup > {SKIP_ABOVE_SECS:.0}s\n");

    let device = match try_device() {
        Some(d) => d,
        None => {
            eprintln!("no wgpu device available -- aborting");
            std::process::exit(1);
        }
    };

    let et = et_config();
    let rf = rf_config();

    for &n in CELL_COUNTS {
        let l = label(n);
        println!("--- shape {n} cells x {N_FEATURES} feats x {N_TARGETS} targets ---");

        let x = make_features(n, SEED as u64 + n as u64);
        let axes = make_targets(&x, n, SEED as u64 + n as u64 + 1);

        run_cpu(n, &x, &axes, &et, &format!("et_cpu_{l}"));
        run_gpu(n, &x, &axes, &et, &device, &format!("et_gpu_{l}"));

        run_cpu(n, &x, &axes, &rf, &format!("rf_cpu_{l}"));
        run_gpu(n, &x, &axes, &rf, &device, &format!("rf_gpu_{l}"));
        println!();
    }
}
