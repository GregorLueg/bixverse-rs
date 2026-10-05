//! GPU-vs-CPU parity for the SCENIC tree learners and GRN entry points.
//!
//! Tree ensembles are seeded but the GPU walks nodes breadth-first where the
//! CPU goes depth-first, so the two RNG streams expose features to nodes in a
//! different order and exact agreement is not on the table. Every fidelity
//! gate here is therefore statistical: top-10 feature overlap or per-target
//! Pearson of the importances, and where the ensemble is small enough to still
//! be noisy, anchored against a CPU-vs-CPU run on a different seed. The GPU
//! passes if it agrees with the CPU at least about as well as the CPU agrees
//! with itself.
//!
//! Sections: toy-shape ExtraTrees and RandomForest checks that run in CI, the
//! large-shape Pearson gates behind `large-test`, a pure-integer check of the
//! coarse threshold widening, and round trips through the GRN entry points.

#![allow(clippy::needless_range_loop, clippy::field_reassign_with_default)]
#![cfg(all(feature = "single-cell", feature = "gpu"))]

use std::collections::HashSet;

use cubecl::Runtime;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use rand::prelude::*;
use rand::rngs::SmallRng;

use bixverse_rs::gpu::sc_gpu::scenic_gpu::{
    ScenicGpuParams, fit_extra_trees_gpu_single, fit_multi_trees_gpu, run_scenic_grn_gpu,
    run_scenic_grn_in_memory_gpu,
};
use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_analysis::scenic::{
    ExtraTreesConfig, GradientBoostingConfig, RandomForestConfig, RegressionLearner, ScenicParams,
    SparseYBatch, fit_multi_trees_sparse,
};
use bixverse_rs::single_cell::sc_data::data_io::CellGeneSparseWriter;
use bixverse_rs::single_cell::sc_traits::F16;
use bixverse_rs::single_cell::sc_utils::utils_tree::QuantisedStore;
#[cfg(feature = "large-test")]
use bixverse_rs::{
    gpu::sc_gpu::scenic_gpu::run_scenic_grn_streaming_gpu,
    single_cell::mc_analysis::scenic_metacells::run_scenic_grn_in_memory,
    single_cell::sc_analysis::scenic::run_scenic_grn,
};

///////////////
// Constants //
///////////////

/// Samples (cells) in the toy shape.
const N_SAMPLES: usize = 256;
/// Features (TFs) in the toy shape.
const N_FEATURES: usize = 32;
/// Targets in the toy shape.
const N_TARGETS: usize = 4;
/// Probability a toy target is non-zero in a given cell.
const SPARSITY: f32 = 0.5;
/// Size of the top-ranked feature lists compared between runs.
const TOP_K: usize = 10;
/// Features `0..N_INFORMATIVE` carry signal for every target. Enough of them
/// that the top-10 list is real signal rather than whichever noise features
/// the DFS and BFS RNG streams happened to draw; otherwise the overlap
/// collapses to near random.
const N_INFORMATIVE: usize = 10;

///////////////////////
// Toy-shape helpers //
///////////////////////

/// Toy [QuantisedStore] with uniform pseudo-random u8 bins.
///
/// ### Params
///
/// * `seed` - Seed for reproducibility
///
/// ### Returns
///
/// An `N_SAMPLES x N_FEATURES` store.
fn make_toy_quantised(seed: u64) -> QuantisedStore {
    let mut rng = SmallRng::seed_from_u64(seed);
    let data: Vec<u8> = (0..N_SAMPLES * N_FEATURES).map(|_| rng.random()).collect();
    QuantisedStore::from_raw(data, N_SAMPLES, N_FEATURES)
}

/// Toy sparse targets, in both the CPU and the GPU layout.
///
/// Every target is a weighted linear combination of the first
/// `N_INFORMATIVE` feature columns with target-specific weights, plus light
/// noise, so features `0..N_INFORMATIVE` are the true top drivers of every
/// target.
///
/// ### Params
///
/// * `x` - The toy feature store
/// * `seed` - Seed for sparsity and noise. The weights use a fixed seed.
///
/// ### Returns
///
/// `(sparse_y, axes)`, the GPU-side [SparseYBatch] and the CPU-side targets,
/// holding identical values.
fn make_toy_targets(x: &QuantisedStore, seed: u64) -> (SparseYBatch, Vec<SparseAxis<u32, f32>>) {
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut cols_indices: Vec<Vec<usize>> = vec![Vec::new(); N_TARGETS];
    let mut cols_values: Vec<Vec<f32>> = vec![Vec::new(); N_TARGETS];

    // Per-target weights on the informative feature block, seeded off a
    // fixed constant so target signal is stable across the seed loop.
    let mut weight_rng = SmallRng::seed_from_u64(0xDEAD_BEEF);
    let mut weights = vec![vec![0.0f32; N_INFORMATIVE]; N_TARGETS];
    for w_row in weights.iter_mut() {
        for w in w_row.iter_mut() {
            *w = weight_rng.random::<f32>() + 0.3;
        }
    }

    let feats: Vec<&[u8]> = (0..N_INFORMATIVE).map(|f| x.get_col(f)).collect();

    for c in 0..N_SAMPLES {
        for t in 0..N_TARGETS {
            if rng.random::<f32>() < SPARSITY {
                let mut signal = 0.0f32;
                for f in 0..N_INFORMATIVE {
                    signal += weights[t][f] * (feats[f][c] as f32 / 255.0);
                }
                let noise: f32 = rng.random::<f32>() * 0.05;
                let v = signal + noise + 0.01;
                cols_indices[t].push(c);
                cols_values[t].push(v);
            }
        }
    }

    let mut axes = Vec::with_capacity(N_TARGETS);
    for t in 0..N_TARGETS {
        axes.push(SparseAxis::<u32, f32>::new_csc(
            cols_indices[t].clone(),
            Vec::new(),
            Some(cols_values[t].clone()),
            N_SAMPLES,
        ));
    }

    // Mirror the private `SparseYBatch::from_targets` layout so both paths
    // see identical sparse Y.
    let mut counts_per_cell = vec![0u32; N_SAMPLES];
    for t in 0..N_TARGETS {
        for &idx in &cols_indices[t] {
            counts_per_cell[idx] += 1;
        }
    }
    let mut offsets = Vec::with_capacity(N_SAMPLES + 1);
    offsets.push(0u32);
    let mut running = 0u32;
    for &c in &counts_per_cell {
        running += c;
        offsets.push(running);
    }
    let total_nnz = running as usize;
    let mut target_indices = vec![0u8; total_nnz];
    let mut values = vec![0.0f32; total_nnz];
    let mut cursor = vec![0u32; N_SAMPLES];
    for (t, (indices, vs)) in cols_indices.iter().zip(cols_values.iter()).enumerate() {
        for (i, &cell) in indices.iter().enumerate() {
            let pos = (offsets[cell] + cursor[cell]) as usize;
            target_indices[pos] = t as u8;
            values[pos] = vs[i];
            cursor[cell] += 1;
        }
    }

    let sparse_y = SparseYBatch {
        offsets,
        target_indices,
        values,
    };

    (sparse_y, axes)
}

/// Indices of the `k` largest importances.
///
/// ### Params
///
/// * `imp` - Importance per feature
/// * `k` - Number of indices to keep
///
/// ### Returns
///
/// Feature indices, largest importance first.
fn top_k_indices(imp: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..imp.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        imp[b]
            .partial_cmp(&imp[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx
}

/// Size of the intersection of two index lists.
///
/// ### Params
///
/// * `a` - First list
/// * `b` - Second list
///
/// ### Returns
///
/// How many entries of `b` are also in `a`.
fn overlap(a: &[usize], b: &[usize]) -> usize {
    let sa: HashSet<usize> = a.iter().copied().collect();
    b.iter().filter(|x| sa.contains(x)).count()
}

/// Importance per feature, summed across targets.
///
/// ### Params
///
/// * `imp` - Importances, indexed `[target][feature]`
///
/// ### Returns
///
/// A length-`N_FEATURES` vector.
fn sum_importances(imp: &[Vec<f32>]) -> Vec<f32> {
    let mut out = vec![0.0f32; N_FEATURES];
    for t in 0..N_TARGETS {
        for f in 0..N_FEATURES {
            out[f] += imp[t][f];
        }
    }
    out
}

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

/// Single-tree ExtraTrees config for the toy shape.
fn config() -> ExtraTreesConfig {
    let mut c = ExtraTreesConfig::default();
    c.n_trees = 1;
    c.max_depth = Some(6);
    // n_thresholds > 1 gives ET a fair chance to find a workable split per
    // feature; keeping min_samples_leaf low (relative to the default 50)
    // lets max_depth = 6 actually build the full tree on 256 samples.
    c.n_thresholds = 5;
    c.min_samples_leaf = 8;
    c.n_features_split = 16;
    c
}

/// RandomForest config at the toy shape, sized to run in a debug CI build.
///
/// RF draws no random thresholds, so it converges far faster than ET and a
/// handful of trees is enough for the informative block to dominate the top-10.
/// Seeds and tree count are kept low deliberately: the CI gpu job builds in
/// debug and falls back to lavapipe software Vulkan on Linux.
///
/// ### Params
///
/// * `n_trees` - Number of trees
///
/// ### Returns
///
/// The populated [RandomForestConfig].
fn rf_toy_config(n_trees: usize) -> RandomForestConfig {
    let mut c = RandomForestConfig::default();
    c.n_trees = n_trees;
    c.max_depth = Some(6);
    // Matches `config()`: the default of 50 would make max_depth 6
    // unreachable on 256 samples.
    c.min_samples_leaf = 8;
    c.n_features_split = 16;
    c
}

/// Pearson correlation of two equal-length vectors.
///
/// ### Params
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// ### Returns
///
/// The correlation, or 0.0 if either vector is constant.
fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let mut num = 0.0f32;
    let mut da = 0.0f32;
    let mut db = 0.0f32;
    for i in 0..a.len() {
        let xa = a[i] - ma;
        let xb = b[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if da <= 0.0 || db <= 0.0 {
        0.0
    } else {
        num / (da * db).sqrt()
    }
}

/////////////////////
// Toy-shape tests //
/////////////////////

/// Statistical-parity check: GPU vs CPU top-10 overlap must be at least
/// as high as the CPU-vs-CPU seed-variance baseline. Any lower would mean
/// the GPU pipeline introduces noise beyond what BFS-vs-DFS RNG ordering
/// already causes.
#[test]
fn test_et_gpu_matches_cpu_top10() {
    let Some(device) = try_device() else { return };

    let cfg = config();

    let mut cpu_gpu_overlaps: Vec<usize> = Vec::new();
    let mut cpu_cpu_overlaps: Vec<usize> = Vec::new();

    for seed_i in 0..10u64 {
        let seed = 20260707 + seed_i;
        let seed_b = seed.wrapping_add(0xC0FFEE);
        let x = make_toy_quantised(seed);
        let (sparse_y, axes) = make_toy_targets(&x, seed.wrapping_add(1));

        let cpu = fit_multi_trees_sparse(&axes, &x, N_SAMPLES, &cfg, seed as usize)
            .expect("CPU fit failed");

        let cpu_b = fit_multi_trees_sparse(&axes, &x, N_SAMPLES, &cfg, seed_b as usize)
            .expect("CPU baseline fit failed");

        let gpu = fit_extra_trees_gpu_single::<WgpuRuntime>(
            &sparse_y,
            &x,
            N_SAMPLES,
            &cfg,
            seed as usize,
            device.clone(),
            &ScenicGpuParams::default(),
        )
        .expect("GPU fit failed");

        assert_eq!(cpu.len(), N_TARGETS);
        assert_eq!(gpu.len(), N_TARGETS);
        for t in 0..N_TARGETS {
            assert_eq!(cpu[t].len(), N_FEATURES);
            assert_eq!(gpu[t].len(), N_FEATURES);
        }

        let cpu_sum = sum_importances(&cpu);
        let cpu_b_sum = sum_importances(&cpu_b);
        let gpu_sum = sum_importances(&gpu);

        let cpu_gpu_ov = overlap(
            &top_k_indices(&cpu_sum, TOP_K),
            &top_k_indices(&gpu_sum, TOP_K),
        );
        let cpu_cpu_ov = overlap(
            &top_k_indices(&cpu_sum, TOP_K),
            &top_k_indices(&cpu_b_sum, TOP_K),
        );
        cpu_gpu_overlaps.push(cpu_gpu_ov);
        cpu_cpu_overlaps.push(cpu_cpu_ov);
    }

    let cpu_gpu_mean =
        cpu_gpu_overlaps.iter().sum::<usize>() as f32 / (cpu_gpu_overlaps.len() * TOP_K) as f32;
    let cpu_cpu_mean =
        cpu_cpu_overlaps.iter().sum::<usize>() as f32 / (cpu_cpu_overlaps.len() * TOP_K) as f32;

    // Sanity floor 1: the GPU path is not obviously broken (well above the
    // 10/32 = 0.31 random baseline).
    assert!(
        cpu_gpu_mean >= 0.32,
        "cpu-gpu top-{TOP_K} overlap {cpu_gpu_mean:.2} at or below random baseline (0.31)"
    );

    // Sanity floor 2: the GPU path is as consistent with CPU as CPU is with
    // itself under a seed change, less a small tolerance to absorb the fact
    // that DFS-vs-BFS RNG stream mismatch is a slightly worse source of
    // disagreement than a fresh CPU seed. Both quantities are subject to
    // per-seed noise -- we only assert on the 10-seed mean.
    assert!(
        cpu_gpu_mean + 0.05 >= cpu_cpu_mean,
        "cpu-gpu top-{TOP_K} overlap {cpu_gpu_mean:.2} materially worse than \
         cpu-cpu seed-variance baseline {cpu_cpu_mean:.2}"
    );
}

/// Statistical-parity check for RandomForest, anchored the same way as
/// [`test_et_gpu_matches_cpu_top10`]: the GPU must agree with the CPU at
/// least as well as two CPU runs on different seeds agree with each other.
///
/// This is the only RF fidelity test that runs in CI. The two 0.95 Pearson
/// gates sit behind `large-test`, which no workflow enables.
#[test]
fn test_rf_gpu_matches_cpu_top10() {
    let Some(device) = try_device() else { return };

    const RF_TOY_TREES: usize = 16;
    const RF_TOY_SEEDS: u64 = 5;

    let cfg = rf_toy_config(RF_TOY_TREES);

    let mut cpu_gpu_overlaps: Vec<usize> = Vec::new();
    let mut cpu_cpu_overlaps: Vec<usize> = Vec::new();

    for seed_i in 0..RF_TOY_SEEDS {
        let seed = 20260711 + seed_i;
        let seed_b = seed.wrapping_add(0xC0FFEE);
        let x = make_toy_quantised(seed);
        let (_sparse_y, axes) = make_toy_targets(&x, seed.wrapping_add(1));

        let cpu = fit_multi_trees_sparse(&axes, &x, N_SAMPLES, &cfg, seed as usize)
            .expect("CPU RF fit failed");
        let cpu_b = fit_multi_trees_sparse(&axes, &x, N_SAMPLES, &cfg, seed_b as usize)
            .expect("CPU RF baseline fit failed");
        let gpu = fit_multi_trees_gpu::<WgpuRuntime>(
            &axes,
            &x,
            N_SAMPLES,
            &cfg,
            seed as usize,
            device.clone(),
            &ScenicGpuParams::default(),
        )
        .expect("GPU RF fit failed");

        assert_eq!(cpu.len(), N_TARGETS);
        assert_eq!(gpu.len(), N_TARGETS);
        for t in 0..N_TARGETS {
            assert_eq!(cpu[t].len(), N_FEATURES);
            assert_eq!(gpu[t].len(), N_FEATURES);
        }

        let cpu_sum = sum_importances(&cpu);
        let cpu_b_sum = sum_importances(&cpu_b);
        let gpu_sum = sum_importances(&gpu);

        // Non-zero importances. A silently dead kernel returns all zeros, which
        // still produces a plausible-looking top-10 out of an arbitrary
        // tie-break order.
        let gpu_total: f32 = gpu_sum.iter().sum();
        assert!(
            gpu_total > 0.0,
            "seed {seed_i}: GPU RF importances are all zero"
        );

        cpu_gpu_overlaps.push(overlap(
            &top_k_indices(&cpu_sum, TOP_K),
            &top_k_indices(&gpu_sum, TOP_K),
        ));
        cpu_cpu_overlaps.push(overlap(
            &top_k_indices(&cpu_sum, TOP_K),
            &top_k_indices(&cpu_b_sum, TOP_K),
        ));
    }

    let cpu_gpu_mean =
        cpu_gpu_overlaps.iter().sum::<usize>() as f32 / (cpu_gpu_overlaps.len() * TOP_K) as f32;
    let cpu_cpu_mean =
        cpu_cpu_overlaps.iter().sum::<usize>() as f32 / (cpu_cpu_overlaps.len() * TOP_K) as f32;

    assert!(
        cpu_gpu_mean >= 0.32,
        "RF cpu-gpu top-{TOP_K} overlap {cpu_gpu_mean:.2} at or below random baseline (0.31)"
    );
    assert!(
        cpu_gpu_mean + 0.05 >= cpu_cpu_mean,
        "RF cpu-gpu top-{TOP_K} overlap {cpu_gpu_mean:.2} materially worse than \
         cpu-cpu seed-variance baseline {cpu_cpu_mean:.2}"
    );
}

/// The multi-tree dispatch still routes ExtraTrees correctly. Smallest ET
/// config that exercises the dispatch: one wave, one batch, short trees. A
/// clearly positive CPU-vs-GPU Pearson is enough to show it is not broken.
#[test]
fn test_et_multi_tree_dispatch() {
    let Some(device) = try_device() else { return };

    const ET_SAMPLES: usize = 1_000;
    const ET_FEATURES: usize = 80;
    const ET_TARGETS: usize = 4;
    const ET_INFORMATIVE: usize = 8;

    let mut rng_x = SmallRng::seed_from_u64(0xABCD_1234);
    let data: Vec<u8> = (0..ET_SAMPLES * ET_FEATURES)
        .map(|_| rng_x.random())
        .collect();
    let x = QuantisedStore::from_raw(data, ET_SAMPLES, ET_FEATURES);

    let mut rng_y = SmallRng::seed_from_u64(0x1234_ABCD);
    let mut cols_indices: Vec<Vec<usize>> = vec![Vec::new(); ET_TARGETS];
    let mut cols_values: Vec<Vec<f32>> = vec![Vec::new(); ET_TARGETS];
    for c in 0..ET_SAMPLES {
        for t in 0..ET_TARGETS {
            if rng_y.random::<f32>() < 0.5 {
                let mut signal = 0.0f32;
                for f in 0..ET_INFORMATIVE {
                    signal += (x.get_col(f)[c] as f32 / 255.0) * 1.5;
                }
                let noise: f32 = rng_y.random::<f32>() * 0.05;
                cols_indices[t].push(c);
                cols_values[t].push(signal + noise);
            }
        }
    }
    let axes: Vec<SparseAxis<u32, f32>> = cols_indices
        .into_iter()
        .zip(cols_values)
        .map(|(idx, vs)| SparseAxis::<u32, f32>::new_csc(idx, Vec::new(), Some(vs), ET_SAMPLES))
        .collect();

    let mut cfg = ExtraTreesConfig::default();
    cfg.n_trees = 50;
    cfg.max_depth = Some(6);
    cfg.min_samples_leaf = 20;
    cfg.n_features_split = 0;
    cfg.n_thresholds = 1;

    let cpu = fit_multi_trees_sparse(&axes, &x, ET_SAMPLES, &cfg, 7).expect("ET CPU fit failed");
    let gpu = fit_multi_trees_gpu::<WgpuRuntime>(
        &axes,
        &x,
        ET_SAMPLES,
        &cfg,
        7,
        device.clone(),
        &ScenicGpuParams::default(),
    )
    .expect("ET GPU fit failed");

    let mut per_target: Vec<f32> = Vec::with_capacity(ET_TARGETS);
    for t in 0..ET_TARGETS {
        per_target.push(pearson(&cpu[t], &gpu[t]));
    }
    let mean_corr = per_target.iter().sum::<f32>() / per_target.len() as f32;

    // ET at 50 trees is still noisy, so the floor only has to be clearly
    // above zero.
    assert!(
        mean_corr >= 0.4,
        "ET dispatch appears broken after RF was added: mean pearson r = {mean_corr:.3}"
    );
}

/////////////////////////
// Large-shape harness //
/////////////////////////

/// Samples in the large shape.
#[cfg(any(feature = "large-test", feature = "large_scale_diagnostics"))]
const LG_N_SAMPLES: usize = 10_000;
/// Features in the large shape.
#[cfg(any(feature = "large-test", feature = "large_scale_diagnostics"))]
const LG_N_FEATURES: usize = 500;
/// Targets in the large shape.
#[cfg(any(feature = "large-test", feature = "large_scale_diagnostics"))]
const LG_N_TARGETS: usize = 20;
/// ExtraTrees ensemble size in the large shape.
#[cfg(any(feature = "large-test", feature = "large_scale_diagnostics"))]
const LG_N_TREES: usize = 500;
/// Informative features. Few of them, so ExtraTrees at 500 trees converges
/// tightly: with 40 spread over 500 features even CPU-vs-CPU only reached
/// ~0.58 Pearson.
#[cfg(any(feature = "large-test", feature = "large_scale_diagnostics"))]
const LG_INFORMATIVE: usize = 10;
/// Probability a large-shape target is non-zero in a given cell.
#[cfg(any(feature = "large-test", feature = "large_scale_diagnostics"))]
const LG_SPARSITY: f32 = 0.5;

/// Large-shape [QuantisedStore] with uniform pseudo-random u8 bins.
///
/// ### Params
///
/// * `seed` - Seed for reproducibility
///
/// ### Returns
///
/// An `LG_N_SAMPLES x LG_N_FEATURES` store.
#[cfg(any(feature = "large-test", feature = "large_scale_diagnostics"))]
fn make_large_quantised(seed: u64) -> QuantisedStore {
    let mut rng = SmallRng::seed_from_u64(seed);
    let data: Vec<u8> = (0..LG_N_SAMPLES * LG_N_FEATURES)
        .map(|_| rng.random())
        .collect();
    QuantisedStore::from_raw(data, LG_N_SAMPLES, LG_N_FEATURES)
}

/// Large-shape sparse targets, built as in [make_toy_targets] but with
/// stronger weights.
///
/// ### Params
///
/// * `x` - The large-shape feature store
/// * `seed` - Seed for sparsity and noise. The weights use a fixed seed.
///
/// ### Returns
///
/// One CSC [SparseAxis] per target.
#[cfg(any(feature = "large-test", feature = "large_scale_diagnostics"))]
fn make_large_targets(x: &QuantisedStore, seed: u64) -> Vec<SparseAxis<u32, f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);

    // Per-target weights on informative features 0..LG_INFORMATIVE, seeded
    // off a fixed constant so target structure is stable across seed changes.
    let mut weight_rng = SmallRng::seed_from_u64(0xF00D_BABE);
    let mut weights = vec![vec![0.0f32; LG_INFORMATIVE]; LG_N_TARGETS];
    for w_row in weights.iter_mut() {
        for w in w_row.iter_mut() {
            // Larger baseline weight = stronger signal per informative
            // feature; ExtraTrees needs signal amplitude well above noise to
            // agree seed-to-seed on which features to rank at the top.
            *w = weight_rng.random::<f32>() + 1.0;
        }
    }

    let feats: Vec<&[u8]> = (0..LG_INFORMATIVE).map(|f| x.get_col(f)).collect();

    let mut cols_indices: Vec<Vec<usize>> = vec![Vec::new(); LG_N_TARGETS];
    let mut cols_values: Vec<Vec<f32>> = vec![Vec::new(); LG_N_TARGETS];
    for c in 0..LG_N_SAMPLES {
        for t in 0..LG_N_TARGETS {
            if rng.random::<f32>() < LG_SPARSITY {
                let mut signal = 0.0f32;
                for f in 0..LG_INFORMATIVE {
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
        .map(|(idx, vs)| SparseAxis::<u32, f32>::new_csc(idx, Vec::new(), Some(vs), LG_N_SAMPLES))
        .collect()
}

///////////////////////
// Large-shape tests //
///////////////////////

/// CPU-vs-CPU baseline at the large shape: is 500 trees enough for the CPU
/// ExtraTrees ensemble to converge? Any GPU-vs-CPU comparison is bounded by
/// this. Prints the mean Pearson, asserts nothing.
#[test]
#[cfg(feature = "large_scale_diagnostics")]
// Heavy: 10k x 500 x 20 with 500 ET trees, two CPU fits.
fn test_et_cpu_seed_baseline_large() {
    let seed_a = 20260708u64;
    let seed_b = seed_a.wrapping_add(0xBEEF);
    let x = make_large_quantised(seed_a);
    let axes = make_large_targets(&x, seed_a.wrapping_add(1));

    let mut cfg = ExtraTreesConfig::default();
    cfg.n_trees = LG_N_TREES;
    cfg.max_depth = Some(10);
    cfg.min_samples_leaf = 50;
    cfg.n_features_split = 0;
    cfg.n_thresholds = 1;

    let cpu_a = fit_multi_trees_sparse(&axes, &x, LG_N_SAMPLES, &cfg, seed_a as usize)
        .expect("CPU A fit failed");
    let cpu_b = fit_multi_trees_sparse(&axes, &x, LG_N_SAMPLES, &cfg, seed_b as usize)
        .expect("CPU B fit failed");

    let mut per_target: Vec<f32> = Vec::with_capacity(LG_N_TARGETS);
    for t in 0..LG_N_TARGETS {
        per_target.push(pearson(&cpu_a[t], &cpu_b[t]));
    }
    let mean_corr = per_target.iter().sum::<f32>() / per_target.len() as f32;
    println!(
        "CPU baseline (n_trees={LG_N_TREES}): mean pearson r = {mean_corr:.3} \
         (per-target: {per_target:?})"
    );
}

/// 500-tree ExtraTrees on 10k x 500 x 20 synthetic data: mean per-target
/// Pearson of the importances against the CPU must clear 0.95. Only sensible
/// in release, where the CPU fit no longer dominates.
#[test]
#[cfg(feature = "large-test")]
// Heavy: 10k x 500 x 20 with 500 ET trees, one CPU fit plus one GPU fit.
fn test_et_gpu_matches_cpu_pearson_large() {
    let Some(device) = try_device() else { return };

    let seed_base = 20260708u64;
    let x = make_large_quantised(seed_base);
    let axes = make_large_targets(&x, seed_base.wrapping_add(1));

    let mut cfg = ExtraTreesConfig::default();
    cfg.n_trees = LG_N_TREES;
    cfg.max_depth = Some(10);
    cfg.min_samples_leaf = 50;
    cfg.n_features_split = 0;
    cfg.n_thresholds = 1;

    let cpu = fit_multi_trees_sparse(&axes, &x, LG_N_SAMPLES, &cfg, seed_base as usize)
        .expect("CPU fit failed");

    let gpu = fit_multi_trees_gpu::<WgpuRuntime>(
        &axes,
        &x,
        LG_N_SAMPLES,
        &cfg,
        seed_base as usize,
        device.clone(),
        &ScenicGpuParams::default(),
    )
    .expect("GPU fit failed");

    assert_eq!(cpu.len(), LG_N_TARGETS);
    assert_eq!(gpu.len(), LG_N_TARGETS);

    let mut per_target: Vec<f32> = Vec::with_capacity(LG_N_TARGETS);
    for t in 0..LG_N_TARGETS {
        assert_eq!(cpu[t].len(), LG_N_FEATURES);
        assert_eq!(gpu[t].len(), LG_N_FEATURES);
        per_target.push(pearson(&cpu[t], &gpu[t]));
    }
    let mean_corr = per_target.iter().sum::<f32>() / per_target.len() as f32;

    assert!(
        mean_corr >= 0.95,
        "ET (large) mean per-target Pearson r = {mean_corr:.3} < 0.95 floor \
         (per-target: {per_target:?})"
    );
}

/// Multi-batch determinism: running one call whose internal chunking
/// produces 3 batches must give byte-identical per-target output to
/// calling `fit_multi_trees_gpu` once per internal batch. Confirms reset
/// semantics for the reused wave state and the tree-seed stream.
///
/// The GPU driver chunks targets at `MULTI_OUTPUT_BATCH = 64`, so 130
/// targets produce three internal batches of (64, 64, 2). We compare
/// against three standalone calls with the same 64/64/2 splits: each
/// standalone call sees the same `n_targets_in_batch` and therefore the
/// same per-tree multi-output scoring, so the trees and importances match.
#[test]
#[cfg(feature = "large-test")]
// Heavy: 2000 x 100 x 130 with 50 trees, so four full GPU ensemble fits.
fn test_multi_batch_matches_chunked() {
    let Some(device) = try_device() else { return };

    // 130 targets -> chunks(64) yields batches of 64, 64, 2
    const MB_N_SAMPLES: usize = 2_000;
    const MB_N_FEATURES: usize = 100;
    const MB_TARGETS: usize = 130;
    const MB_N_TREES: usize = 50;
    const MB_BATCH: usize = 64;

    let x = {
        let mut rng = SmallRng::seed_from_u64(31415);
        let data: Vec<u8> = (0..MB_N_SAMPLES * MB_N_FEATURES)
            .map(|_| rng.random())
            .collect();
        QuantisedStore::from_raw(data, MB_N_SAMPLES, MB_N_FEATURES)
    };
    let axes = {
        let mut rng = SmallRng::seed_from_u64(27182);
        let mut cols_indices: Vec<Vec<usize>> = vec![Vec::new(); MB_TARGETS];
        let mut cols_values: Vec<Vec<f32>> = vec![Vec::new(); MB_TARGETS];
        for c in 0..MB_N_SAMPLES {
            for t in 0..MB_TARGETS {
                if rng.random::<f32>() < 0.3 {
                    let signal = x.get_col(t % 10)[c] as f32 / 255.0;
                    let noise: f32 = rng.random::<f32>() * 0.05;
                    cols_indices[t].push(c);
                    cols_values[t].push(signal + noise);
                }
            }
        }
        cols_indices
            .into_iter()
            .zip(cols_values)
            .map(|(idx, vs)| {
                SparseAxis::<u32, f32>::new_csc(idx, Vec::new(), Some(vs), MB_N_SAMPLES)
            })
            .collect::<Vec<_>>()
    };

    let mut cfg = ExtraTreesConfig::default();
    cfg.n_trees = MB_N_TREES;
    cfg.max_depth = Some(6);
    cfg.min_samples_leaf = 20;
    cfg.n_features_split = 0;
    cfg.n_thresholds = 1;

    let combined = fit_multi_trees_gpu::<WgpuRuntime>(
        &axes,
        &x,
        MB_N_SAMPLES,
        &cfg,
        42,
        device.clone(),
        &ScenicGpuParams::default(),
    )
    .expect("combined GPU fit failed");

    // Standalone calls with the same chunk sizes as the internal split. Same
    // seed means each chunk sees the same tree-seed stream. Same
    // n_targets_in_batch means the same multi-output scoring, hence the
    // same trees.
    let mut split = Vec::with_capacity(MB_TARGETS);
    for chunk in axes.chunks(MB_BATCH) {
        let part = fit_multi_trees_gpu::<WgpuRuntime>(
            chunk,
            &x,
            MB_N_SAMPLES,
            &cfg,
            42,
            device.clone(),
            &ScenicGpuParams::default(),
        )
        .expect("chunked GPU fit failed");
        for v in part {
            split.push(v);
        }
    }

    assert_eq!(combined.len(), split.len());
    let mut max_diff = 0.0f32;
    for t in 0..combined.len() {
        assert_eq!(combined[t].len(), split[t].len());
        for f in 0..combined[t].len() {
            let d = (combined[t][f] - split[t][f]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    assert!(
        max_diff < 1e-5,
        "combined batch fit differs from chunked by max {max_diff:.2e} -- \
         batch grouping is affecting per-tree output"
    );
}

/// 250-tree RandomForest (no bootstrap, subsample_rate=0.632) on the
/// large-shape harness: mean per-target Pearson r against the CPU >= 0.95.
///
/// RF is more expensive than ET per split (exhaustive threshold scan over
/// ~254 candidates per feature vs 1 random threshold), so the tree count is
/// left at the CPU RF default of 250 rather than ET's 500. Run under
/// `cargo test --release` -- debug mode is dominated by CPU-side RF.
#[test]
#[cfg(feature = "large-test")]
// Heavy: 10k x 500 x 20 with 250 RF trees, one CPU fit plus one GPU fit.
fn test_rf_gpu_matches_cpu_pearson_large() {
    let Some(device) = try_device() else { return };

    let seed_base = 20260709u64;
    let x = make_large_quantised(seed_base);
    let axes = make_large_targets(&x, seed_base.wrapping_add(1));

    let mut cfg = RandomForestConfig::default();
    cfg.max_depth = Some(10);
    cfg.min_samples_leaf = 50;
    cfg.n_features_split = 0;
    // n_trees, subsample_rate=0.632, bootstrap=false stay at defaults

    let cpu = fit_multi_trees_sparse(&axes, &x, LG_N_SAMPLES, &cfg, seed_base as usize)
        .expect("CPU RF fit failed");

    let gpu = fit_multi_trees_gpu::<WgpuRuntime>(
        &axes,
        &x,
        LG_N_SAMPLES,
        &cfg,
        seed_base as usize,
        device.clone(),
        &ScenicGpuParams::default(),
    )
    .expect("GPU RF fit failed");

    let mut per_target: Vec<f32> = Vec::with_capacity(LG_N_TARGETS);
    for t in 0..LG_N_TARGETS {
        per_target.push(pearson(&cpu[t], &gpu[t]));
    }
    let mean_corr = per_target.iter().sum::<f32>() / per_target.len() as f32;

    assert!(
        mean_corr >= 0.95,
        "RF (large, no bootstrap) mean per-target Pearson r = {mean_corr:.3} < 0.95"
    );
}

/// Bootstrap variant of [test_rf_gpu_matches_cpu_pearson_large]: same RF config
/// with bootstrap-with-replacement enabled, same 0.95 floor.
#[test]
#[cfg(feature = "large-test")]
// Heavy: 10k x 500 x 20 with 250 RF trees, one CPU fit plus one GPU fit.
fn test_rf_bootstrap_gpu_matches_cpu_pearson_large() {
    let Some(device) = try_device() else { return };

    let seed_base = 20260710u64;
    let x = make_large_quantised(seed_base);
    let axes = make_large_targets(&x, seed_base.wrapping_add(1));

    let mut cfg = RandomForestConfig::default();
    cfg.max_depth = Some(10);
    cfg.min_samples_leaf = 50;
    cfg.n_features_split = 0;
    cfg.bootstrap = true;

    let cpu = fit_multi_trees_sparse(&axes, &x, LG_N_SAMPLES, &cfg, seed_base as usize)
        .expect("CPU RF+bootstrap fit failed");

    let gpu = fit_multi_trees_gpu::<WgpuRuntime>(
        &axes,
        &x,
        LG_N_SAMPLES,
        &cfg,
        seed_base as usize,
        device.clone(),
        &ScenicGpuParams::default(),
    )
    .expect("GPU RF+bootstrap fit failed");

    let mut per_target: Vec<f32> = Vec::with_capacity(LG_N_TARGETS);
    for t in 0..LG_N_TARGETS {
        per_target.push(pearson(&cpu[t], &gpu[t]));
    }
    let mean_corr = per_target.iter().sum::<f32>() / per_target.len() as f32;

    assert!(
        mean_corr >= 0.95,
        "RF (large, bootstrap) mean per-target Pearson r = {mean_corr:.3} < 0.95"
    );
}

/// Reduced-shape RF Pearson gate.
///
/// [test_rf_gpu_matches_cpu_pearson_large] is the real fidelity gate, but at
/// 10k cells / 250 trees it is far too slow for a debug build on lavapipe. This
/// runs the same comparison at a fraction of the work.
///
/// Two assertions, and the second is the one that matters. An absolute floor
/// alone is not meaningful here: what the comparison can reach is bounded by
/// how far the ensemble has converged, and at this shape a CPU run against
/// another CPU run on a different seed only reaches ~0.966 itself. So the test
/// also anchors against that baseline, which is what distinguishes "the GPU
/// diverged" from "120 trees is not many trees".
///
/// Calibrated, not guessed: measured 0.878 cpu-gpu against a 0.891 cpu-cpu
/// baseline at 30 trees, 0.966 against 0.966 at 120, and 0.982 against 0.984
/// at 300. The GPU sits on the noise ceiling at every tree count. 120 is the
/// cheapest of those that leaves headroom over a 0.95 floor, and the result is
/// stable to three decimals across repeat runs despite the CAS-loop atomic in
/// `accumulate_importance` varying summation order.
#[test]
#[cfg(feature = "large-test")]
// Heavy: 1000 x 50 x 8 with 120 RF trees, two CPU fits plus one GPU fit.
fn test_rf_gpu_matches_cpu_pearson_small() {
    let Some(device) = try_device() else { return };

    const RS_N_SAMPLES: usize = 1_000;
    const RS_N_FEATURES: usize = 50;
    const RS_N_TARGETS: usize = 8;
    const RS_N_TREES: usize = 120;
    const RS_INFORMATIVE: usize = 6;
    const RS_SPARSITY: f32 = 0.5;
    const RS_PEARSON_FLOOR: f32 = 0.95;

    let seed_base = 20260712u64;

    let x = {
        let mut rng = SmallRng::seed_from_u64(seed_base);
        let data: Vec<u8> = (0..RS_N_SAMPLES * RS_N_FEATURES)
            .map(|_| rng.random())
            .collect();
        QuantisedStore::from_raw(data, RS_N_SAMPLES, RS_N_FEATURES)
    };

    let axes = {
        let mut rng = SmallRng::seed_from_u64(seed_base.wrapping_add(1));
        // Weights seeded off a fixed constant so target structure is stable if
        // the shape seeds ever move.
        let mut weight_rng = SmallRng::seed_from_u64(0x5EED_1234);
        let mut weights = vec![vec![0.0f32; RS_INFORMATIVE]; RS_N_TARGETS];
        for w_row in weights.iter_mut() {
            for w in w_row.iter_mut() {
                *w = weight_rng.random::<f32>() + 1.0;
            }
        }
        let feats: Vec<&[u8]> = (0..RS_INFORMATIVE).map(|f| x.get_col(f)).collect();

        let mut cols_indices: Vec<Vec<usize>> = vec![Vec::new(); RS_N_TARGETS];
        let mut cols_values: Vec<Vec<f32>> = vec![Vec::new(); RS_N_TARGETS];
        for c in 0..RS_N_SAMPLES {
            for t in 0..RS_N_TARGETS {
                if rng.random::<f32>() < RS_SPARSITY {
                    let mut signal = 0.0f32;
                    for f in 0..RS_INFORMATIVE {
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
            .map(|(idx, vs)| {
                SparseAxis::<u32, f32>::new_csc(idx, Vec::new(), Some(vs), RS_N_SAMPLES)
            })
            .collect::<Vec<_>>()
    };

    let mut cfg = RandomForestConfig::default();
    cfg.n_trees = RS_N_TREES;
    cfg.max_depth = Some(5);
    cfg.min_samples_leaf = 20;
    cfg.n_features_split = 0;

    let cpu = fit_multi_trees_sparse(&axes, &x, RS_N_SAMPLES, &cfg, seed_base as usize)
        .expect("CPU RF fit failed");
    let cpu_b = fit_multi_trees_sparse(
        &axes,
        &x,
        RS_N_SAMPLES,
        &cfg,
        seed_base.wrapping_add(0xBEEF) as usize,
    )
    .expect("CPU RF baseline fit failed");
    let gpu = fit_multi_trees_gpu::<WgpuRuntime>(
        &axes,
        &x,
        RS_N_SAMPLES,
        &cfg,
        seed_base as usize,
        device.clone(),
        &ScenicGpuParams::default(),
    )
    .expect("GPU RF fit failed");

    let mut per_target: Vec<f32> = Vec::with_capacity(RS_N_TARGETS);
    let mut baseline: Vec<f32> = Vec::with_capacity(RS_N_TARGETS);
    for t in 0..RS_N_TARGETS {
        assert_eq!(cpu[t].len(), RS_N_FEATURES);
        assert_eq!(gpu[t].len(), RS_N_FEATURES);
        per_target.push(pearson(&cpu[t], &gpu[t]));
        baseline.push(pearson(&cpu[t], &cpu_b[t]));
    }
    let mean_corr = per_target.iter().sum::<f32>() / per_target.len() as f32;
    let baseline_corr = baseline.iter().sum::<f32>() / baseline.len() as f32;

    assert!(
        mean_corr >= RS_PEARSON_FLOOR,
        "RF (small) cpu-gpu Pearson r = {mean_corr:.3} < {RS_PEARSON_FLOOR} floor \
         (per-target: {per_target:?})"
    );
    assert!(
        mean_corr + 0.05 >= baseline_corr,
        "RF (small) cpu-gpu Pearson r = {mean_corr:.3} materially worse than the \
         cpu-cpu seed-variance baseline {baseline_corr:.3}; the GPU is diverging beyond \
         what the tree count explains"
    );
}

/// RF fidelity when the quantised bins are *skewed*, which is what real data
/// looks like and what none of the other GPU tests exercise.
///
/// Every other test here builds its store with `QuantisedStore::from_raw` and
/// uniform random u8 bins. Uniform bins are the easy case for the GPU's coarser
/// bin axis: merging them loses almost nothing. Real single-cell data quantises
/// to a spike at bin 0 with a long right tail, where the informative split
/// points are crowded into the tail and merging bins there is a much bigger ask.
///
/// This is the test that has to hold before `SMEM_HIST_SLOTS` is tightened, not
/// the uniform ones.
#[test]
#[cfg(feature = "large-test")]
// Heavy: 1500 x 50 x 64 with 120 RF trees, two CPU fits plus one GPU fit.
fn test_rf_gpu_matches_cpu_pearson_skewed_bins() {
    let Some(device) = try_device() else { return };

    const SK_N_SAMPLES: usize = 1_500;
    const SK_N_FEATURES: usize = 50;
    const SK_N_TARGETS: usize = 64; // full batch, so the GPU takes its coarsest bins
    const SK_N_TREES: usize = 120;
    const SK_INFORMATIVE: usize = 6;
    const SK_ZERO_FRAC: f32 = 0.6;

    let seed_base = 20260713u64;

    // Spike at bin 0 plus a right-skewed tail, roughly what QuantisedStore
    // produces from sparse counts.
    let x = {
        let mut rng = SmallRng::seed_from_u64(seed_base);
        let data: Vec<u8> = (0..SK_N_SAMPLES * SK_N_FEATURES)
            .map(|_| {
                if rng.random::<f32>() < SK_ZERO_FRAC {
                    0u8
                } else {
                    let u: f32 = rng.random();
                    (u * u * 255.0) as u8
                }
            })
            .collect();
        QuantisedStore::from_raw(data, SK_N_SAMPLES, SK_N_FEATURES)
    };

    let axes = {
        let mut rng = SmallRng::seed_from_u64(seed_base.wrapping_add(1));
        let mut weight_rng = SmallRng::seed_from_u64(0x51CE_D000);
        let mut weights = vec![vec![0.0f32; SK_INFORMATIVE]; SK_N_TARGETS];
        for w_row in weights.iter_mut() {
            for w in w_row.iter_mut() {
                *w = weight_rng.random::<f32>() + 1.0;
            }
        }
        let feats: Vec<&[u8]> = (0..SK_INFORMATIVE).map(|f| x.get_col(f)).collect();

        let mut cols_indices: Vec<Vec<usize>> = vec![Vec::new(); SK_N_TARGETS];
        let mut cols_values: Vec<Vec<f32>> = vec![Vec::new(); SK_N_TARGETS];
        for c in 0..SK_N_SAMPLES {
            for t in 0..SK_N_TARGETS {
                if rng.random::<f32>() < 0.5 {
                    let mut signal = 0.0f32;
                    for f in 0..SK_INFORMATIVE {
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
            .map(|(idx, vs)| {
                SparseAxis::<u32, f32>::new_csc(idx, Vec::new(), Some(vs), SK_N_SAMPLES)
            })
            .collect::<Vec<_>>()
    };

    let mut cfg = RandomForestConfig::default();
    cfg.n_trees = SK_N_TREES;
    cfg.max_depth = Some(6);
    cfg.min_samples_leaf = 20;
    cfg.n_features_split = 0;

    let cpu = fit_multi_trees_sparse(&axes, &x, SK_N_SAMPLES, &cfg, seed_base as usize)
        .expect("CPU RF fit failed");
    let cpu_b = fit_multi_trees_sparse(
        &axes,
        &x,
        SK_N_SAMPLES,
        &cfg,
        seed_base.wrapping_add(0xBEEF) as usize,
    )
    .expect("CPU RF baseline fit failed");
    let gpu = fit_multi_trees_gpu::<WgpuRuntime>(
        &axes,
        &x,
        SK_N_SAMPLES,
        &cfg,
        seed_base as usize,
        device.clone(),
        &ScenicGpuParams::default(),
    )
    .expect("GPU RF fit failed");

    let mut per_target: Vec<f32> = Vec::with_capacity(SK_N_TARGETS);
    let mut baseline: Vec<f32> = Vec::with_capacity(SK_N_TARGETS);
    for t in 0..SK_N_TARGETS {
        per_target.push(pearson(&cpu[t], &gpu[t]));
        baseline.push(pearson(&cpu[t], &cpu_b[t]));
    }
    let mean_corr = per_target.iter().sum::<f32>() / per_target.len() as f32;
    let baseline_corr = baseline.iter().sum::<f32>() / baseline.len() as f32;

    assert!(
        mean_corr >= 0.95,
        "RF on skewed bins: cpu-gpu Pearson r = {mean_corr:.3} < 0.95 floor \
         (per-target: {per_target:?})"
    );
    assert!(
        mean_corr + 0.05 >= baseline_corr,
        "RF on skewed bins: cpu-gpu Pearson r = {mean_corr:.3} materially worse than the \
         cpu-cpu seed-variance baseline {baseline_corr:.3}"
    );
}

//////////////////////////
// Coarse bin threshold //
//////////////////////////

/// The GPU bins at a coarser resolution than the shared
/// [`QuantisedStore`], which stays at 256 u8 bins. A split found at coarse
/// threshold `thr_c` has to be written back as a fine threshold so
/// `reassign_samples` and the CPU-side tree semantics keep working unchanged.
///
/// The widening is `(thr_c << shift) | ((1 << shift) - 1)`, and this asserts it
/// is exact for every shift, threshold and bin: `b >> shift <= thr_c` must hold
/// for exactly the same bins as `b <= widened`.
#[test]
fn test_coarse_threshold_roundtrip() {
    for shift in 1..=3u32 {
        let n_coarse = 256u32 >> shift;
        // Threshold n_coarse - 1 sends everything left and is never a
        // candidate, matching how the fine path excludes bin 255.
        for thr_c in 0..n_coarse - 1 {
            let widened = (thr_c << shift) | ((1u32 << shift) - 1);
            assert!(
                widened < 256,
                "shift {shift}, thr_c {thr_c}: widened {widened} out of u8 range"
            );
            for b in 0..256u32 {
                assert_eq!(
                    b >> shift <= thr_c,
                    b <= widened,
                    "shift {shift}, thr_c {thr_c}, bin {b}: coarse and widened \
                     tests disagree"
                );
            }
        }
    }
}

/////////////////////////////
// Entry-point round trips //
/////////////////////////////

// Plumbing checks for the GRN entry points against the CPU equivalents. The
// 0.85 floor is lower than the fit kernel's 0.95 because gene batching adds
// its own randomisation on top of the fit noise; what is under test is that
// file I/O, batching and matrix assembly wire up, not the fit itself.

/// Cells in the round-trip fixture.
const RT_CELLS: usize = 200;
/// TFs, genes `0..RT_TFS`.
const RT_TFS: usize = 12;
/// Targets, genes `RT_TFS..RT_TOTAL_GENES`.
const RT_TARGETS: usize = 20;
/// Total genes.
const RT_TOTAL_GENES: usize = RT_TFS + RT_TARGETS;

/// Write a small synthetic gene-based sparse expression file.
///
/// The signal is deliberately strong: for each target `t`, TF `t % RT_TFS`
/// drives the expression; other TFs contribute noise. That ensures the top TF
/// per target is stable across seeds so per-target Pearson clears the
/// threshold in reasonable wall-clock.
///
/// ### Params
///
/// * `path` - Where to write the file
fn write_synthetic_scenic_file(path: &str) {
    let mut writer =
        CellGeneSparseWriter::new(path, false, RT_CELLS, RT_TOTAL_GENES, 1e4).expect("writer new");

    let mut rng = SmallRng::seed_from_u64(0xA5A5);

    // Sample once per (cell, TF) so every target uses the same TF profile.
    let tf_cell_vals: Vec<Vec<f32>> = (0..RT_TFS)
        .map(|_| {
            (0..RT_CELLS)
                .map(|_| rng.random_range(0.0..10.0f32))
                .collect()
        })
        .collect();

    for tf in 0..RT_TFS {
        write_gene_chunk_from_dense(&mut writer, tf, &tf_cell_vals[tf]);
    }

    for tg in 0..RT_TARGETS {
        let driver_tf = tg % RT_TFS;
        let vals: Vec<f32> = (0..RT_CELLS)
            .map(|c| {
                let driver = tf_cell_vals[driver_tf][c];
                let noise: f32 = rng.random_range(-1.0..1.0);
                (2.0 * driver + noise).max(0.0)
            })
            .collect();
        write_gene_chunk_from_dense(&mut writer, RT_TFS + tg, &vals);
    }

    writer.finalise().expect("writer finalise");
}

/// Writes one gene from a dense per-cell vector, rounding to u16 raw counts
/// and storing `ln(1 + v)` as the normalised layer.
///
/// ### Params
///
/// * `writer` - Open writer
/// * `gene_id` - Gene index
/// * `vals` - Expression per cell; non-positive or round-to-zero values are dropped
fn write_gene_chunk_from_dense(writer: &mut CellGeneSparseWriter, gene_id: usize, vals: &[f32]) {
    let mut data_raw: Vec<u16> = Vec::new();
    let mut data_norm: Vec<F16> = Vec::new();
    let mut indices: Vec<usize> = Vec::new();
    for (cell_idx, &v) in vals.iter().enumerate() {
        if v <= 0.0 {
            continue;
        }
        let raw = v.round().clamp(0.0, u16::MAX as f32) as u16;
        if raw == 0 {
            continue;
        }
        data_raw.push(raw);
        data_norm.push(F16::from(half::f16::from_f32(v.ln_1p())));
        indices.push(cell_idx);
    }
    let raw = RawCounts::U16(data_raw);
    let chunk = CscGeneChunk::from_conversion(raw, &data_norm, &indices, gene_id, true);
    writer.write_gene_chunk(chunk).expect("write chunk");
}

/// Row-wise Pearson between two equal-shape matrices.
///
/// ### Params
///
/// * `cpu` - Reference, targets x TFs
/// * `gpu` - Comparison, same shape
///
/// ### Returns
///
/// One correlation per row.
#[cfg(feature = "large-test")]
fn pearson_per_target(cpu: faer::MatRef<f32>, gpu: faer::MatRef<f32>) -> Vec<f32> {
    let n_targets = cpu.nrows();
    let n_features = cpu.ncols();
    assert_eq!(n_targets, gpu.nrows());
    assert_eq!(n_features, gpu.ncols());
    (0..n_targets)
        .map(|t| {
            let a: Vec<f32> = (0..n_features).map(|j| *cpu.get(t, j)).collect();
            let b: Vec<f32> = (0..n_features).map(|j| *gpu.get(t, j)).collect();
            pearson(&a, &b)
        })
        .collect()
}

/// SCENIC params for the round trips: 100 ET trees, random gene batches of 8,
/// no cell or gene filtering.
fn scenic_params_for_roundtrip() -> ScenicParams {
    let mut cfg = ExtraTreesConfig::default();
    cfg.n_trees = 100;
    cfg.max_depth = Some(6);
    cfg.min_samples_leaf = 20;
    cfg.n_features_split = 0;
    cfg.n_thresholds = 1;

    ScenicParams {
        min_counts: 0,
        min_cells: 0.0,
        regression_learner: RegressionLearner::ExtraTrees(cfg),
        gene_batch_strategy: "random".to_string(),
        gene_batch_size: Some(8),
        n_pcs: 5,
        n_subsample: RT_CELLS,
    }
}

/// Reader-backed GRN: GPU must track CPU at mean per-target Pearson >= 0.85.
#[test]
#[cfg(feature = "large-test")]
// Heavy: 200 x 32 full reader pipeline, CPU and GPU, fixture on disk.
fn test_run_scenic_grn_gpu_roundtrip() {
    let Some(device) = try_device() else { return };

    let path = std::env::temp_dir().join("bixverse_scenic_gpu_roundtrip.bin");
    let path_str = path.to_str().expect("utf-8 temp path");
    write_synthetic_scenic_file(path_str);

    let cell_indices: Vec<usize> = (0..RT_CELLS).collect();
    let tf_indices: Vec<usize> = (0..RT_TFS).collect();
    let gene_indices: Vec<usize> = (RT_TFS..RT_TOTAL_GENES).collect();

    let params = scenic_params_for_roundtrip();

    let reader = ParallelSparseReader::new(path_str).expect("failed to open synthetic file");

    let cpu = run_scenic_grn(
        &reader,
        &cell_indices,
        &gene_indices,
        &tf_indices,
        &params,
        42,
        0,
    )
    .expect("CPU run_scenic_grn failed");

    let gpu = run_scenic_grn_gpu::<WgpuRuntime, _>(
        &reader,
        &cell_indices,
        &gene_indices,
        &tf_indices,
        &params,
        &ScenicGpuParams::default(),
        42,
        device,
        0,
    )
    .expect("GPU run_scenic_grn_gpu failed");

    let corrs = pearson_per_target(cpu.as_ref(), gpu.as_ref());
    let mean = corrs.iter().sum::<f32>() / corrs.len() as f32;
    assert!(
        mean >= 0.85,
        "mean per-target Pearson {mean:.3} below 0.85 threshold: {corrs:?}"
    );

    let _ = std::fs::remove_file(&path);
}

/// Same gate for the streaming path, which chunks genes off disk.
#[test]
#[cfg(feature = "large-test")]
// Heavy: 200 x 32 full streaming pipeline, CPU and GPU, fixture on disk.
fn test_run_scenic_grn_streaming_gpu_roundtrip() {
    let Some(device) = try_device() else { return };

    let path = std::env::temp_dir().join("bixverse_scenic_gpu_streaming_roundtrip.bin");
    let path_str = path.to_str().expect("utf-8 temp path");
    write_synthetic_scenic_file(path_str);

    let cell_indices: Vec<usize> = (0..RT_CELLS).collect();
    let tf_indices: Vec<usize> = (0..RT_TFS).collect();
    let gene_indices: Vec<usize> = (RT_TFS..RT_TOTAL_GENES).collect();

    let params = scenic_params_for_roundtrip();

    let reader = ParallelSparseReader::new(path_str).expect("failed to open synthetic file");

    let cpu = run_scenic_grn(
        &reader,
        &cell_indices,
        &gene_indices,
        &tf_indices,
        &params,
        42,
        0,
    )
    .expect("CPU baseline failed");

    let gpu = run_scenic_grn_streaming_gpu::<WgpuRuntime, _>(
        &reader,
        &cell_indices,
        &gene_indices,
        &tf_indices,
        &params,
        &ScenicGpuParams::default(),
        42,
        device,
        0,
    )
    .expect("GPU streaming failed");

    let corrs = pearson_per_target(cpu.as_ref(), gpu.as_ref());
    let mean = corrs.iter().sum::<f32>() / corrs.len() as f32;
    assert!(
        mean >= 0.85,
        "mean per-target Pearson {mean:.3} below 0.85 threshold: {corrs:?}"
    );

    let _ = std::fs::remove_file(&path);
}

/// Small in-memory CSC for the metacell round trip. Same design as the disk
/// fixture: TF `t % RT_TFS` drives target `t`, so the top-TF-per-target signal
/// is stable and the Pearson threshold clears in seconds.
///
/// ### Returns
///
/// A `RT_CELLS x RT_TOTAL_GENES` CSC matrix with raw u16 and f32 layers.
fn build_synthetic_scenic_csc() -> CompressedSparseData2<u16, f32> {
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    let mut data: Vec<u16> = Vec::new();
    let mut data_2: Vec<f32> = Vec::new();
    let mut indices: Vec<usize> = Vec::new();
    let mut indptr: Vec<usize> = vec![0];

    // TF cell profiles, retained so target columns can be derived from them.
    let mut tf_cell_vals: Vec<Vec<f32>> = vec![vec![0.0f32; RT_CELLS]; RT_TFS];
    for tf in 0..RT_TFS {
        for c in 0..RT_CELLS {
            let v: f32 = rng.random_range(0.0..10.0);
            if v > 0.5 {
                indices.push(c);
                data.push(v as u16);
                data_2.push(v);
                tf_cell_vals[tf][c] = v;
            }
        }
        indptr.push(data.len());
    }

    for tg in 0..RT_TARGETS {
        let driver_tf = tg % RT_TFS;
        for c in 0..RT_CELLS {
            let driver = tf_cell_vals[driver_tf][c];
            let noise: f32 = rng.random_range(-1.0..1.0);
            let v = (2.0 * driver + noise).max(0.0);
            if v > 0.5 {
                indices.push(c);
                data.push(v as u16);
                data_2.push(v);
            }
        }
        indptr.push(data.len());
    }

    CompressedSparseData2 {
        data,
        indices: indices.index_cast(),
        indptr: indptr.index_cast(),
        cs_type: CompressedSparseFormat::Csc,
        data_2: Some(data_2),
        shape: (RT_CELLS, RT_TOTAL_GENES),
    }
}

/// Same gate for the in-memory CSC path, comparing the target rows only.
#[test]
#[cfg(feature = "large-test")]
// Heavy: 200 x 32 full in-memory pipeline, CPU and GPU.
fn test_run_scenic_grn_in_memory_gpu_roundtrip() {
    let Some(device) = try_device() else { return };

    let csc = build_synthetic_scenic_csc();
    let tf_indices: Vec<usize> = (0..RT_TFS).collect();

    let params = scenic_params_for_roundtrip();

    let cpu = run_scenic_grn_in_memory(&csc, &tf_indices, &params, 42, 0)
        .expect("CPU run_scenic_grn_in_memory failed");

    let gpu = run_scenic_grn_in_memory_gpu::<WgpuRuntime, u16>(
        &csc,
        &tf_indices,
        &params,
        &ScenicGpuParams::default(),
        42,
        device,
        0,
    )
    .expect("GPU run_scenic_grn_in_memory_gpu failed");

    // CPU returns (n_total_genes, n_tfs), so restrict to the target rows.
    let cpu_targets = cpu.as_ref().submatrix(RT_TFS, 0, RT_TARGETS, RT_TFS);
    let gpu_targets = gpu.as_ref().submatrix(RT_TFS, 0, RT_TARGETS, RT_TFS);
    let corrs = pearson_per_target(cpu_targets, gpu_targets);
    let mean = corrs.iter().sum::<f32>() / corrs.len() as f32;
    assert!(
        mean >= 0.85,
        "mean per-target Pearson {mean:.3} below 0.85 threshold: {corrs:?}"
    );
}

/// Gradient boosting has no GPU learner: must error, not fall back.
#[test]
fn test_run_scenic_grn_in_memory_gpu_rejects_gbm() {
    let Some(device) = try_device() else { return };

    let csc = build_synthetic_scenic_csc();
    let tf_indices: Vec<usize> = (0..RT_TFS).collect();

    let mut params = scenic_params_for_roundtrip();
    params.regression_learner =
        RegressionLearner::GradientBoosting(GradientBoostingConfig::default());

    let err = run_scenic_grn_in_memory_gpu::<WgpuRuntime, u16>(
        &csc,
        &tf_indices,
        &params,
        &ScenicGpuParams::default(),
        42,
        device,
        0,
    )
    .expect_err("expected GpuNotSupportedForLearner");

    match err {
        BixverseErrors::GpuNotSupportedForLearner { learner } => {
            assert_eq!(learner, "GradientBoosting");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

/// The same rejection on the reader path, ahead of any I/O.
#[test]
fn test_run_scenic_grn_gpu_rejects_gbm() {
    let Some(device) = try_device() else { return };

    // The file has to exist: the learner check happens inside
    // `run_scenic_grn_gpu`, so a real reader has to be built first.
    let path = std::env::temp_dir().join("bixverse_scenic_gpu_gbm_reject.bin");
    let path_str = path.to_str().expect("utf-8 temp path");
    write_synthetic_scenic_file(path_str);

    let cell_indices: Vec<usize> = (0..RT_CELLS).collect();
    let tf_indices: Vec<usize> = (0..RT_TFS).collect();
    let gene_indices: Vec<usize> = (RT_TFS..RT_TOTAL_GENES).collect();

    let mut params = scenic_params_for_roundtrip();
    params.regression_learner =
        RegressionLearner::GradientBoosting(GradientBoostingConfig::default());

    let reader = ParallelSparseReader::new(path_str).expect("failed to open synthetic file");

    let err = run_scenic_grn_gpu::<WgpuRuntime, _>(
        &reader,
        &cell_indices,
        &gene_indices,
        &tf_indices,
        &params,
        &ScenicGpuParams::default(),
        42,
        device,
        0,
    )
    .expect_err("expected GpuNotSupportedForLearner");

    match err {
        BixverseErrors::GpuNotSupportedForLearner { learner } => {
            assert_eq!(learner, "GradientBoosting");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }

    let _ = std::fs::remove_file(&path);
}
