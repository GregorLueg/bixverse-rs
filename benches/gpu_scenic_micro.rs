//! Tight-loop CPU vs GPU harness for the SCENIC tree regressor.
//!
//! The full `gpu_scenic_bench` takes hours. This one takes about a minute and
//! measures the single number that decides whether a kernel change was worth
//! keeping: **GPU wall clock against one sequential CPU core**.
//!
//! `fit_multi_trees_sparse` is sequential over trees, while the end-to-end CPU
//! driver fans out over gene batches with rayon. So the GPU has to beat one
//! core by more than the machine's effective core count before the end-to-end
//! comparison can flip. On a 10-core M1 Max that means roughly 10x.
//!
//! Shape mirrors one end-to-end batch exactly (10k cells, 1000 TFs, 64
//! targets, depth 10, `min_samples_leaf` 50). Only `n_trees` shrinks. Two tree
//! counts run per learner: [`TREES_ONE_WAVE`] fits in a single wave, so the gap
//! to [`TREES_MULTI_WAVE`] separates fixed per-batch cost (`WaveState`
//! allocation, feature upload, sparse Y upload) from per-wave kernel cost.
//!
//! A final multi-batch cell runs [`N_BATCHES`] gene batches. The single-batch
//! cells structurally flatter the CPU, because the GPU pays its whole fixed
//! per-call cost to serve one batch while production serves ~63 from one call.
//! The multi-batch ratio is the one to trust when estimating end-to-end.
//!
//! Run with:
//! ```
//! cargo bench --features gpu,single-cell --bench gpu_scenic_micro
//! ```
//!
//! Set `SCENIC_MICRO_CELLS=50000` to run the larger guard shape instead. The
//! archive table shows the GPU ratio improving with cell count, so 10k is its
//! worst shape and worth optimising against, but check 50k at milestones so we
//! do not tune into that corner.

#![cfg(all(feature = "gpu", feature = "single-cell"))]
#![allow(clippy::field_reassign_with_default, clippy::needless_range_loop)]

use std::time::Instant;

use bixverse_rs::gpu::sc_gpu::scenic_gpu::{
    ScenicGpuParams, fit_multi_trees_gpu, pick_wave_size, viable_max_active_nodes, wave_byte_cost,
    wave_byte_cost_et,
};
use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_analysis::scenic::{
    ExtraTreesConfig, RandomForestConfig, TreeRegressorConfig, fit_multi_trees_sparse,
};
use bixverse_rs::single_cell::sc_utils::utils_tree::{QuantisedStore, resolve_n_features_split};

use cubecl::Runtime;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use rand::prelude::*;
use rand::rngs::SmallRng;

////////////
// Shapes //
////////////

/// Default cell count. Matches the shape the end-to-end matrix was measured
/// at, which is also where the GPU looks worst.
const DEFAULT_CELLS: usize = 10_000;

/// TF count, i.e. the feature axis. `n_features_split = 0` resolves this to
/// `sqrt(1000) = 31` features drawn per node.
const N_FEATURES: usize = 1_000;

/// Targets per batch. `MULTI_OUTPUT_BATCH` is 64, so this is exactly one
/// end-to-end gene batch.
const N_TARGETS: usize = 64;

/// Single-wave tree count. `DEFAULT_WAVE_SIZE` is 8, so this is one wave and
/// its wall clock is dominated by fixed per-batch cost.
const TREES_ONE_WAVE: usize = 8;

/// Multi-wave tree count. Four waves at the current default, so
/// `(t_multi - t_one) / 3` is the marginal cost of a wave.
const TREES_MULTI_WAVE: usize = 32;

/// Tree depth, matching the end-to-end config.
const MAX_DEPTH: usize = 10;

/// Minimum samples per leaf. Also caps `viable_max_active_nodes` at
/// `n_samples / 100`, which is what bounds the histogram allocation.
const MIN_SAMPLES_LEAF: usize = 50;

/// Fraction of (cell, target) pairs that carry a nonzero value.
const SPARSITY: f32 = 0.5;

/// Features that actually drive the synthetic targets.
const N_INFORMATIVE: usize = 10;

/// Fixed seed so successive runs compare like with like.
const SEED: usize = 20260708;

/// 12 GiB, matching `gpu_scenic_bench`. Deliberately not the 4 GiB library
/// default: the point is to reproduce the conditions the archive table was
/// measured under, where the wave lands at 8 rather than 4.
const WAVE_BUDGET: usize = 12 * 1024 * 1024 * 1024;

/// Measured repeats after one warmup. Minimum is reported, which is more
/// robust to background load than a mean over so few samples.
const N_REPEATS: usize = 2;

/// Trees used for the warmup call. Enough to exercise the wave loop while
/// staying cheap; its only real job is forcing shader compilation before the
/// clock starts.
const WARMUP_TREES: usize = 2;

/// Gene batches in the multi-batch cell. The single-batch cells pay the full
/// fixed per-call cost (feature upload, `WaveState` allocation, readback) over
/// one batch, which structurally flatters the CPU: production runs ~63 batches
/// through one `fit_scenic_batches_gpu` call that uploads features once and
/// defers every readback. Four batches is enough to see the fixed cost
/// amortise without the CPU side taking all afternoon.
const N_BATCHES: usize = 4;

/// Trees for the multi-batch cell. Lower than [`TREES_MULTI_WAVE`] because the
/// CPU side runs `N_BATCHES` sequential fits and would otherwise dominate the
/// harness runtime.
const TREES_MULTI_BATCH: usize = 16;

////////////////
// Data build //
////////////////

/// Build a random quantised feature store.
///
/// Deliberately uniform random bins rather than anything realistic. Split
/// quality is irrelevant here; what matters is that every kernel does the same
/// volume of work it would on real data.
///
/// ### Params
///
/// * `n_samples` - Cell count.
/// * `seed` - RNG seed.
///
/// ### Returns
///
/// A `QuantisedStore` of `n_samples x N_FEATURES` u8 bins, column-major.
fn make_features(n_samples: usize, seed: u64) -> QuantisedStore {
    let mut rng = SmallRng::seed_from_u64(seed);
    let data: Vec<u8> = (0..n_samples * N_FEATURES).map(|_| rng.random()).collect();
    QuantisedStore::from_raw(data, n_samples, N_FEATURES)
}

/// Build synthetic sparse target columns driven by the first `N_INFORMATIVE`
/// features, so the trees find real structure and the split path is exercised
/// rather than bailing out on zero variance.
///
/// ### Params
///
/// * `x` - Feature store the targets are generated from.
/// * `n_samples` - Cell count.
/// * `n_targets` - Columns to generate. Use a multiple of [`N_TARGETS`] to get
///   whole gene batches out of the driver's internal chunking.
/// * `seed` - RNG seed.
///
/// ### Returns
///
/// `n_targets` sparse columns, each with roughly `SPARSITY * n_samples`
/// nonzeros.
fn make_targets(
    x: &QuantisedStore,
    n_samples: usize,
    n_targets: usize,
    seed: u64,
) -> Vec<SparseAxis<u32, f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut weight_rng = SmallRng::seed_from_u64(0xF00D_BABE);
    let mut weights = vec![vec![0.0f32; N_INFORMATIVE]; n_targets];
    for w_row in weights.iter_mut() {
        for w in w_row.iter_mut() {
            *w = weight_rng.random::<f32>() + 1.0;
        }
    }

    let feats: Vec<&[u8]> = (0..N_INFORMATIVE).map(|f| x.get_col(f)).collect();

    let mut cols_indices: Vec<Vec<usize>> = vec![Vec::new(); n_targets];
    let mut cols_values: Vec<Vec<f32>> = vec![Vec::new(); n_targets];
    for c in 0..n_samples {
        for t in 0..n_targets {
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

/////////////
// Configs //
/////////////

/// ExtraTrees config at the given tree count. `n_thresholds` stays at its
/// default of 1, which is the whole reason the bin axis of the histogram is
/// dead weight on this path.
fn et_config(n_trees: usize) -> ExtraTreesConfig {
    let mut c = ExtraTreesConfig::default();
    c.n_trees = n_trees;
    c.max_depth = Some(MAX_DEPTH);
    c.min_samples_leaf = MIN_SAMPLES_LEAF;
    c.n_features_split = 0;
    c
}

/// RandomForest config at the given tree count. Exhaustive threshold search,
/// so it genuinely consumes the whole cumulative histogram.
fn rf_config(n_trees: usize) -> RandomForestConfig {
    let mut c = RandomForestConfig::default();
    c.n_trees = n_trees;
    c.max_depth = Some(MAX_DEPTH);
    c.min_samples_leaf = MIN_SAMPLES_LEAF;
    c.n_features_split = 0;
    c
}

/// Open the default wgpu device, or `None` when no adapter is available.
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

/// Run `f` once to warm up, then [`N_REPEATS`] times, returning the fastest
/// measured run in seconds.
fn best_of<F: FnMut()>(mut f: F) -> f32 {
    f();
    let mut best = f32::MAX;
    for _ in 0..N_REPEATS {
        let t0 = Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f32());
    }
    best
}

/// Time one (learner, tree count) cell on both backends and print the ratio.
///
/// The GPU is warmed at [`WARMUP_TREES`] rather than `n_trees` so shader
/// compilation lands outside the measured region without paying for a full
/// extra run.
///
/// ### Params
///
/// * `label` - Row label, e.g. `"et"`.
/// * `n_samples` - Cell count.
/// * `x` - Feature store.
/// * `axes` - Target columns.
/// * `config` - Tree config at the measured tree count.
/// * `warmup` - Same learner at [`WARMUP_TREES`] trees.
/// * `device` - wgpu device.
fn run_cell(
    label: &str,
    n_samples: usize,
    x: &QuantisedStore,
    axes: &[SparseAxis<u32, f32>],
    config: &dyn TreeRegressorConfig,
    warmup: &dyn TreeRegressorConfig,
    device: &WgpuDevice,
) {
    let params = ScenicGpuParams {
        wave_byte_budget: WAVE_BUDGET,
    };

    let cpu = best_of(|| {
        fit_multi_trees_sparse(axes, x, n_samples, config, SEED).expect("CPU fit failed");
    });

    fit_multi_trees_gpu::<WgpuRuntime>(axes, x, n_samples, warmup, SEED, device.clone(), &params)
        .expect("GPU warmup failed");

    let mut gpu = f32::MAX;
    let mut checksum = 0.0f64;
    for _ in 0..N_REPEATS {
        let t0 = Instant::now();
        let imp = fit_multi_trees_gpu::<WgpuRuntime>(
            axes,
            x,
            n_samples,
            config,
            SEED,
            device.clone(),
            &params,
        )
        .expect("GPU fit failed");
        gpu = gpu.min(t0.elapsed().as_secs_f32());
        checksum = imp.iter().flatten().map(|&v| v as f64).sum();
    }

    // Silent-failure guard. The kernels launch via `launch_unchecked`, so a
    // binding that busts a device limit produces no work and no error, just an
    // implausibly fast run returning zeros. Normalised importances sum to 1.0
    // per target, so a healthy run checksums to n_targets.
    assert!(
        checksum > 0.5 * axes.len() as f64,
        "{label}: importance checksum {checksum:.3} vs expected ~{}; \
         the GPU almost certainly did no work",
        axes.len()
    );

    println!(
        "  {label:<18} cpu {cpu:>7.2}s  gpu {gpu:>7.2}s  ratio {:>6.2}x",
        cpu / gpu
    );
}

/// Time [`N_BATCHES`] gene batches on both backends and print the ratio.
///
/// The two sides are deliberately asymmetric, mirroring production. The CPU
/// runs one sequential `fit_multi_trees_sparse` per batch, which is exactly
/// what a single rayon worker does in `run_scenic_multi_output_in_memory`. The
/// GPU gets one `fit_multi_trees_gpu` call over all the targets, which chunks
/// internally at `MULTI_OUTPUT_BATCH` and so uploads the feature tensor once
/// and defers every importance readback.
///
/// ### Params
///
/// * `label` - Row label.
/// * `n_samples` - Cell count.
/// * `x` - Feature store.
/// * `axes` - All `N_BATCHES * N_TARGETS` target columns.
/// * `config` - Tree config at [`TREES_MULTI_BATCH`] trees.
/// * `warmup` - Same learner at [`WARMUP_TREES`] trees.
/// * `device` - wgpu device.
fn run_multi_batch_cell(
    label: &str,
    n_samples: usize,
    x: &QuantisedStore,
    axes: &[SparseAxis<u32, f32>],
    config: &dyn TreeRegressorConfig,
    warmup: &dyn TreeRegressorConfig,
    device: &WgpuDevice,
) {
    let params = ScenicGpuParams {
        wave_byte_budget: WAVE_BUDGET,
    };

    let cpu = best_of(|| {
        for chunk in axes.chunks(N_TARGETS) {
            fit_multi_trees_sparse(chunk, x, n_samples, config, SEED).expect("CPU fit failed");
        }
    });

    fit_multi_trees_gpu::<WgpuRuntime>(axes, x, n_samples, warmup, SEED, device.clone(), &params)
        .expect("GPU warmup failed");

    let mut gpu = f32::MAX;
    let mut checksum = 0.0f64;
    for _ in 0..N_REPEATS {
        let t0 = Instant::now();
        let imp = fit_multi_trees_gpu::<WgpuRuntime>(
            axes,
            x,
            n_samples,
            config,
            SEED,
            device.clone(),
            &params,
        )
        .expect("GPU fit failed");
        gpu = gpu.min(t0.elapsed().as_secs_f32());
        checksum = imp.iter().flatten().map(|&v| v as f64).sum();
    }

    assert!(
        checksum > 0.5 * axes.len() as f64,
        "{label}: importance checksum {checksum:.3} vs expected ~{}; \
         the GPU almost certainly did no work",
        axes.len()
    );

    println!(
        "  {label:<18} cpu {cpu:>7.2}s  gpu {gpu:>7.2}s  ratio {:>6.2}x",
        cpu / gpu
    );
}

/// Print the wave scheduler's decision for this shape, so the VRAM story stays
/// visible as the histogram allocations change.
fn report_shape(n_samples: usize, device: &WgpuDevice) {
    let max_binding = WgpuRuntime::client(device)
        .properties()
        .memory
        .max_page_size as usize;
    let k_feats = resolve_n_features_split(0, N_FEATURES).clamp(1, N_FEATURES);
    let nodes = viable_max_active_nodes(MAX_DEPTH, n_samples, MIN_SAMPLES_LEAF);
    let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);

    let wave_rf = pick_wave_size(
        nodes,
        k_feats,
        N_TARGETS,
        TREES_MULTI_WAVE,
        WAVE_BUDGET,
        1,
        false,
        max_binding,
    )
    .expect("shape busts the wave budget at wave_size = 1");
    let wave_et = pick_wave_size(
        nodes,
        k_feats,
        N_TARGETS,
        TREES_MULTI_WAVE,
        WAVE_BUDGET,
        1,
        true,
        max_binding,
    )
    .expect("shape busts the wave budget at wave_size = 1");

    println!(
        "  shape: {n_samples} cells, {N_FEATURES} TFs, {N_TARGETS} targets, \
         k_feats {k_feats}, max_active_nodes {nodes}"
    );
    println!(
        "  wave:  et size {wave_et} ({:.3} GiB) | rf size {wave_rf} ({:.2} GiB) | \
         budget {:.0} GiB | max binding {:.2} GiB",
        gib(wave_byte_cost_et(wave_et, nodes, k_feats, N_TARGETS, 1)),
        gib(wave_byte_cost(wave_rf, nodes, k_feats, N_TARGETS)),
        gib(WAVE_BUDGET),
        gib(max_binding),
    );
}

fn main() {
    let n_samples: usize = std::env::var("SCENIC_MICRO_CELLS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_CELLS);

    let Some(device) = try_device() else {
        eprintln!("no wgpu device available -- aborting");
        std::process::exit(1);
    };

    println!("gpu_scenic_micro: best of {N_REPEATS} after 1 warmup");
    report_shape(n_samples, &device);
    println!("  target: ratio >= 10x (M1 Max has ~10 cores for the rayon fan-out)\n");

    let x = make_features(n_samples, SEED as u64 + n_samples as u64);
    let axes = make_targets(
        &x,
        n_samples,
        N_TARGETS * N_BATCHES,
        SEED as u64 + n_samples as u64 + 1,
    );
    let one_batch = &axes[..N_TARGETS];

    let et_warm = et_config(WARMUP_TREES);
    let rf_warm = rf_config(WARMUP_TREES);

    for &n_trees in &[TREES_ONE_WAVE, TREES_MULTI_WAVE] {
        println!("--- {n_trees} trees ---");
        run_cell(
            &format!("et_{n_trees}t"),
            n_samples,
            &x,
            one_batch,
            &et_config(n_trees),
            &et_warm,
            &device,
        );
        run_cell(
            &format!("rf_{n_trees}t"),
            n_samples,
            &x,
            one_batch,
            &rf_config(n_trees),
            &rf_warm,
            &device,
        );
        println!();
    }

    println!("--- {N_BATCHES} batches x {N_TARGETS} targets, {TREES_MULTI_BATCH} trees ---");
    run_multi_batch_cell(
        "et_multibatch",
        n_samples,
        &x,
        &axes,
        &et_config(TREES_MULTI_BATCH),
        &et_warm,
        &device,
    );
    run_multi_batch_cell(
        "rf_multibatch",
        n_samples,
        &x,
        &axes,
        &rf_config(TREES_MULTI_BATCH),
        &rf_warm,
        &device,
    );
}
