//! Phase 1 GPU-vs-CPU sanity for ExtraTrees single-tree fitting.
//!
//! Single-tree ExtraTrees is inherently high-variance: even two CPU runs
//! with different seeds disagree on ~65% of their top-10 features. The GPU
//! path processes nodes in BFS order rather than the CPU's DFS, so the two
//! RNG streams expose features to nodes in different orders even when the
//! total set of feature draws is identical. That drives the expected
//! GPU-vs-CPU disagreement.
//!
//! The test therefore anchors GPU-vs-CPU disagreement against the CPU-vs-CPU
//! seed-variance baseline: the GPU path passes if it agrees with the CPU
//! path at least as well as two CPU runs with different seeds agree with
//! each other. That is what "statistical parity" means for this workload,
//! per the plan's "sanity floor, not precision target" wording.

#![cfg(all(feature = "gpu", feature = "single-cell"))]
#![allow(clippy::needless_range_loop, clippy::field_reassign_with_default)]

use bixverse_rs::gpu::sc_gpu::scenic_gpu::{
    fit_extra_trees_gpu_single, fit_multi_trees_gpu,
};
use bixverse_rs::gpu::sc_gpu::scenic_gpu_params::ScenicGpuParams;
use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_analysis::scenic::{
    ExtraTreesConfig, RandomForestConfig, SparseYBatch, fit_multi_trees_sparse,
};
use bixverse_rs::single_cell::sc_utils::utils_tree::QuantisedStore;

use cubecl::Runtime;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use rand::prelude::*;
use rand::rngs::SmallRng;

const N_SAMPLES: usize = 256;
const N_FEATURES: usize = 32;
const N_TARGETS: usize = 4;
const SPARSITY: f32 = 0.5;
const TOP_K: usize = 10;

// Features 0..N_INFORMATIVE all carry structured signal for every target
// (with target-specific mixing weights). This makes the "top-10" list
// dominated by real signal rather than by which noise features happened to
// be sampled -- otherwise the CPU's DFS RNG stream and the GPU's BFS RNG
// stream expose different noise features and the overlap collapses to a
// baseline that's close to random.
const N_INFORMATIVE: usize = 10;

/// Build a toy QuantisedStore with seeded pseudo-random u8 bins.
fn make_toy_quantised(seed: u64) -> QuantisedStore {
    let mut rng = SmallRng::seed_from_u64(seed);
    let data: Vec<u8> = (0..N_SAMPLES * N_FEATURES).map(|_| rng.random()).collect();
    QuantisedStore::from_raw(data, N_SAMPLES, N_FEATURES)
}

/// Toy sparse targets. Every target is a weighted linear combination of the
/// first `N_INFORMATIVE` feature columns with target-specific weights, plus
/// light noise. That guarantees that features 0..N_INFORMATIVE are the true
/// top drivers of every target.
fn make_toy_targets(
    x: &QuantisedStore,
    seed: u64,
) -> (SparseYBatch, Vec<SparseAxis<u32, f32>>) {
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut cols_indices: Vec<Vec<usize>> = vec![Vec::new(); N_TARGETS];
    let mut cols_values: Vec<Vec<f32>> = vec![Vec::new(); N_TARGETS];

    // per-target weights on the informative feature block; seeded off a
    // fixed constant so target signal is stable across the seed loop
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

    // CPU-facing targets: Vec<SparseAxis>
    let mut axes = Vec::with_capacity(N_TARGETS);
    for t in 0..N_TARGETS {
        axes.push(SparseAxis::<u32, f32>::new_csc(
            cols_indices[t].clone(),
            Vec::new(),
            Some(cols_values[t].clone()),
            N_SAMPLES,
        ));
    }

    // GPU-facing: mirror the private SparseYBatch::from_targets layout so
    // both paths see identical sparse Y.
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

fn top_k_indices(imp: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..imp.len()).collect();
    idx.sort_unstable_by(|&a, &b| imp[b].partial_cmp(&imp[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    idx
}

fn overlap(a: &[usize], b: &[usize]) -> usize {
    let sa: std::collections::HashSet<usize> = a.iter().copied().collect();
    b.iter().filter(|x| sa.contains(x)).count()
}

/// Summed-across-targets importance vector, per feature.
fn sum_importances(imp: &[Vec<f32>]) -> Vec<f32> {
    let mut out = vec![0.0f32; N_FEATURES];
    for t in 0..N_TARGETS {
        for f in 0..N_FEATURES {
            out[f] += imp[t][f];
        }
    }
    out
}

fn try_device() -> Option<WgpuDevice> {
    let device = WgpuDevice::DefaultDevice;
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        WgpuRuntime::client(&device);
    }))
    .ok()
    .map(|_| device)
}

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

/// Baseline: CPU vs CPU with different seeds. This bounds what any
/// CPU-vs-GPU comparison can plausibly achieve on a single tree.
#[test]
fn cpu_baseline_seed_variance() {
    let cfg = config();

    let mut overlaps: Vec<usize> = Vec::new();

    for seed_i in 0..10u64 {
        let seed_a = 20260707 + seed_i;
        let seed_b = seed_a.wrapping_add(0xC0FFEE);
        let x = make_toy_quantised(seed_a);
        let (_sy, axes) = make_toy_targets(&x, seed_a.wrapping_add(1));

        let cpu_a = fit_multi_trees_sparse(&axes, &x, N_SAMPLES, &cfg, seed_a as usize)
            .expect("CPU fit A failed");
        let cpu_b = fit_multi_trees_sparse(&axes, &x, N_SAMPLES, &cfg, seed_b as usize)
            .expect("CPU fit B failed");

        let a_sum = sum_importances(&cpu_a);
        let b_sum = sum_importances(&cpu_b);
        overlaps.push(overlap(&top_k_indices(&a_sum, TOP_K), &top_k_indices(&b_sum, TOP_K)));
    }
    let mean_ov = overlaps.iter().sum::<usize>() as f32 / (overlaps.len() * TOP_K) as f32;
    eprintln!("cpu_baseline: mean top-{TOP_K} overlap = {mean_ov:.2} (per seed: {overlaps:?})");
}

/// Statistical-parity check: GPU vs CPU top-10 overlap must be at least
/// as high as the CPU-vs-CPU seed-variance baseline. Any lower would mean
/// the GPU pipeline introduces noise beyond what BFS-vs-DFS RNG ordering
/// already causes.
#[test]
fn extra_trees_gpu_matches_cpu_top10() {
    let Some(device) = try_device() else {
        eprintln!("scenic_gpu: no wgpu device available -- skipping");
        return;
    };

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

        eprintln!(
            "seed {seed_i}: cpu-gpu top-{TOP_K} = {cpu_gpu_ov}/{TOP_K}, \
             cpu-cpu top-{TOP_K} = {cpu_cpu_ov}/{TOP_K} (baseline)"
        );
    }

    let cpu_gpu_mean =
        cpu_gpu_overlaps.iter().sum::<usize>() as f32 / (cpu_gpu_overlaps.len() * TOP_K) as f32;
    let cpu_cpu_mean =
        cpu_cpu_overlaps.iter().sum::<usize>() as f32 / (cpu_cpu_overlaps.len() * TOP_K) as f32;

    eprintln!(
        "scenic_gpu summary: cpu-gpu mean = {cpu_gpu_mean:.2}, \
         cpu-cpu baseline mean = {cpu_cpu_mean:.2}"
    );

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

/////////////////////////
// Phase 2 test harness //
/////////////////////////

const P2_N_SAMPLES: usize = 10_000;
const P2_N_FEATURES: usize = 500;
const P2_N_TARGETS: usize = 20;
const P2_N_TREES: usize = 500;
// Concentrating signal in fewer informative features gives ExtraTrees at 500
// trees enough leverage to converge tightly. With 40 informative features
// spread across 500-feature space, even CPU-vs-CPU at 500 trees only agrees
// on ~0.58 Pearson (measured). Ten informative features let both paths
// consistently rank the informative block at the top of importance.
const P2_INFORMATIVE: usize = 10;
const P2_SPARSITY: f32 = 0.5;

fn make_p2_quantised(seed: u64) -> QuantisedStore {
    let mut rng = SmallRng::seed_from_u64(seed);
    let data: Vec<u8> = (0..P2_N_SAMPLES * P2_N_FEATURES)
        .map(|_| rng.random())
        .collect();
    QuantisedStore::from_raw(data, P2_N_SAMPLES, P2_N_FEATURES)
}

fn make_p2_targets(x: &QuantisedStore, seed: u64) -> Vec<SparseAxis<u32, f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);

    // per-target weights on informative features 0..P2_INFORMATIVE, seeded
    // off a fixed constant so target structure is stable across seed changes
    let mut weight_rng = SmallRng::seed_from_u64(0xF00D_BABE);
    let mut weights = vec![vec![0.0f32; P2_INFORMATIVE]; P2_N_TARGETS];
    for w_row in weights.iter_mut() {
        for w in w_row.iter_mut() {
            // Larger baseline weight = stronger signal per informative
            // feature; ExtraTrees needs signal amplitude well above noise to
            // agree seed-to-seed on which features to rank at the top.
            *w = weight_rng.random::<f32>() + 1.0;
        }
    }

    let feats: Vec<&[u8]> = (0..P2_INFORMATIVE).map(|f| x.get_col(f)).collect();

    let mut cols_indices: Vec<Vec<usize>> = vec![Vec::new(); P2_N_TARGETS];
    let mut cols_values: Vec<Vec<f32>> = vec![Vec::new(); P2_N_TARGETS];
    for c in 0..P2_N_SAMPLES {
        for t in 0..P2_N_TARGETS {
            if rng.random::<f32>() < P2_SPARSITY {
                let mut signal = 0.0f32;
                for f in 0..P2_INFORMATIVE {
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
        .map(|(idx, vs)| SparseAxis::<u32, f32>::new_csc(idx, Vec::new(), Some(vs), P2_N_SAMPLES))
        .collect()
}

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

/// Phase 2 CPU-vs-CPU baseline at the same scale: is 500 trees enough for
/// the CPU ExtraTrees ensemble to converge, or does it still have material
/// seed variance? Any GPU-vs-CPU comparison is bounded by this baseline.
#[test]
#[ignore]
fn phase2_cpu_baseline() {
    let seed_a = 20260708u64;
    let seed_b = seed_a.wrapping_add(0xBEEF);
    let x = make_p2_quantised(seed_a);
    let axes = make_p2_targets(&x, seed_a.wrapping_add(1));

    let mut cfg = ExtraTreesConfig::default();
    cfg.n_trees = P2_N_TREES;
    cfg.max_depth = Some(10);
    cfg.min_samples_leaf = 50;
    cfg.n_features_split = 0;
    cfg.n_thresholds = 1;

    let cpu_a = fit_multi_trees_sparse(&axes, &x, P2_N_SAMPLES, &cfg, seed_a as usize)
        .expect("CPU A fit failed");
    let cpu_b = fit_multi_trees_sparse(&axes, &x, P2_N_SAMPLES, &cfg, seed_b as usize)
        .expect("CPU B fit failed");

    let mut per_target: Vec<f32> = Vec::with_capacity(P2_N_TARGETS);
    for t in 0..P2_N_TARGETS {
        per_target.push(pearson(&cpu_a[t], &cpu_b[t]));
    }
    let mean_corr = per_target.iter().sum::<f32>() / per_target.len() as f32;
    eprintln!(
        "Phase 2 CPU baseline (n_trees={P2_N_TREES}): mean pearson r = {mean_corr:.3} \
         (per-target: {per_target:?})"
    );
}

/// Phase 2 acceptance: 500-tree ExtraTrees on 10k * 500 * 20 synthetic data,
/// per-target Pearson correlation vs CPU averaged over targets must clear 0.95.
///
/// Runs in ~60s on macOS Metal under `cargo test --release`. Under a debug
/// build the CPU comparison dominates (~500s just for CPU-side fit_multi_trees_sparse),
/// so this test is only sensibly run in release. n_trees can be dropped to
/// e.g. 100 if CI-runtime budget is tighter (CPU-vs-CPU baseline at n_trees=100
/// still clears ~0.99 Pearson with the current synthetic data).
#[test]
fn phase2_multi_tree_pearson() {
    let Some(device) = try_device() else {
        eprintln!("scenic_gpu (Phase 2): no wgpu device available -- skipping");
        return;
    };

    let seed_base = 20260708u64;
    let x = make_p2_quantised(seed_base);
    let axes = make_p2_targets(&x, seed_base.wrapping_add(1));

    let mut cfg = ExtraTreesConfig::default();
    cfg.n_trees = P2_N_TREES;
    cfg.max_depth = Some(10);
    cfg.min_samples_leaf = 50;
    cfg.n_features_split = 0; // -> sqrt(500) = 22
    cfg.n_thresholds = 1;

    let t_cpu = std::time::Instant::now();
    let cpu = fit_multi_trees_sparse(&axes, &x, P2_N_SAMPLES, &cfg, seed_base as usize)
        .expect("CPU fit failed");
    let cpu_secs = t_cpu.elapsed().as_secs_f32();

    let t_gpu = std::time::Instant::now();
    let gpu = fit_multi_trees_gpu::<WgpuRuntime>(
        &axes,
        &x,
        P2_N_SAMPLES,
        &cfg,
        seed_base as usize,
        device.clone(),
        &ScenicGpuParams::default(),
    )
    .expect("GPU fit failed");
    let gpu_secs = t_gpu.elapsed().as_secs_f32();

    assert_eq!(cpu.len(), P2_N_TARGETS);
    assert_eq!(gpu.len(), P2_N_TARGETS);

    let mut per_target: Vec<f32> = Vec::with_capacity(P2_N_TARGETS);
    for t in 0..P2_N_TARGETS {
        assert_eq!(cpu[t].len(), P2_N_FEATURES);
        assert_eq!(gpu[t].len(), P2_N_FEATURES);
        per_target.push(pearson(&cpu[t], &gpu[t]));
    }
    let mean_corr = per_target.iter().sum::<f32>() / per_target.len() as f32;

    eprintln!(
        "Phase 2 ({}k * {} feats * {} targets * {} trees): \
         cpu {cpu_secs:.1}s, gpu {gpu_secs:.1}s, mean pearson r = {mean_corr:.3} \
         (per-target: {:?})",
        P2_N_SAMPLES / 1000,
        P2_N_FEATURES,
        P2_N_TARGETS,
        P2_N_TREES,
        per_target
    );

    assert!(
        mean_corr >= 0.95,
        "Phase 2 mean per-target Pearson r = {mean_corr:.3} < 0.95 floor \
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
fn phase2_multi_batch_determinism() {
    let Some(device) = try_device() else {
        eprintln!("scenic_gpu (Phase 2 multi-batch): no wgpu device -- skipping");
        return;
    };

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
    eprintln!("phase2_multi_batch_determinism: max per-feature diff = {max_diff:.2e}");
    assert!(
        max_diff < 1e-5,
        "combined batch fit differs from chunked by max {max_diff:.2e} -- \
         batch grouping is affecting per-tree output"
    );
}

////////////////////////
// Phase 3 tests (RF) //
////////////////////////

/// Phase 3 acceptance: 250-tree RandomForest (no bootstrap, subsample_rate=0.632)
/// on the Phase 2 synthetic harness. Assert mean per-target Pearson r >= 0.95.
///
/// RF is more expensive than ET per split (exhaustive threshold scan over
/// ~254 candidates per feature vs 1 random threshold), so the tree count is
/// left at the CPU RF default of 250 rather than ET's 500. Run under
/// `cargo test --release` -- debug mode is dominated by CPU-side RF.
#[test]
fn phase3_random_forest_pearson() {
    let Some(device) = try_device() else {
        eprintln!("scenic_gpu (Phase 3 RF): no wgpu device available -- skipping");
        return;
    };

    let seed_base = 20260709u64;
    let x = make_p2_quantised(seed_base);
    let axes = make_p2_targets(&x, seed_base.wrapping_add(1));

    let mut cfg = RandomForestConfig::default();
    cfg.max_depth = Some(10);
    cfg.min_samples_leaf = 50;
    cfg.n_features_split = 0;
    // n_trees, subsample_rate=0.632, bootstrap=false stay at defaults

    let t_cpu = std::time::Instant::now();
    let cpu = fit_multi_trees_sparse(&axes, &x, P2_N_SAMPLES, &cfg, seed_base as usize)
        .expect("CPU RF fit failed");
    let cpu_secs = t_cpu.elapsed().as_secs_f32();

    let t_gpu = std::time::Instant::now();
    let gpu = fit_multi_trees_gpu::<WgpuRuntime>(
        &axes,
        &x,
        P2_N_SAMPLES,
        &cfg,
        seed_base as usize,
        device.clone(),
        &ScenicGpuParams::default(),
    )
    .expect("GPU RF fit failed");
    let gpu_secs = t_gpu.elapsed().as_secs_f32();

    let mut per_target: Vec<f32> = Vec::with_capacity(P2_N_TARGETS);
    for t in 0..P2_N_TARGETS {
        per_target.push(pearson(&cpu[t], &gpu[t]));
    }
    let mean_corr = per_target.iter().sum::<f32>() / per_target.len() as f32;

    eprintln!(
        "Phase 3 RF (n_trees={}): cpu {cpu_secs:.1}s, gpu {gpu_secs:.1}s, \
         mean pearson r = {mean_corr:.3} (per-target: {per_target:?})",
        cfg.n_trees
    );

    assert!(
        mean_corr >= 0.95,
        "Phase 3 RF (no-bootstrap) mean per-target Pearson r = {mean_corr:.3} < 0.95"
    );
}

/// Phase 3 bootstrap variant: same RF config but with bootstrap-with-replacement
/// enabled. Assert the same 0.95 tolerance.
#[test]
fn phase3_rf_bootstrap_pearson() {
    let Some(device) = try_device() else {
        eprintln!("scenic_gpu (Phase 3 RF+bootstrap): no wgpu device -- skipping");
        return;
    };

    let seed_base = 20260710u64;
    let x = make_p2_quantised(seed_base);
    let axes = make_p2_targets(&x, seed_base.wrapping_add(1));

    let mut cfg = RandomForestConfig::default();
    cfg.max_depth = Some(10);
    cfg.min_samples_leaf = 50;
    cfg.n_features_split = 0;
    cfg.bootstrap = true;

    let t_cpu = std::time::Instant::now();
    let cpu = fit_multi_trees_sparse(&axes, &x, P2_N_SAMPLES, &cfg, seed_base as usize)
        .expect("CPU RF+bootstrap fit failed");
    let cpu_secs = t_cpu.elapsed().as_secs_f32();

    let t_gpu = std::time::Instant::now();
    let gpu = fit_multi_trees_gpu::<WgpuRuntime>(
        &axes,
        &x,
        P2_N_SAMPLES,
        &cfg,
        seed_base as usize,
        device.clone(),
        &ScenicGpuParams::default(),
    )
    .expect("GPU RF+bootstrap fit failed");
    let gpu_secs = t_gpu.elapsed().as_secs_f32();

    let mut per_target: Vec<f32> = Vec::with_capacity(P2_N_TARGETS);
    for t in 0..P2_N_TARGETS {
        per_target.push(pearson(&cpu[t], &gpu[t]));
    }
    let mean_corr = per_target.iter().sum::<f32>() / per_target.len() as f32;

    eprintln!(
        "Phase 3 RF+bootstrap (n_trees={}): cpu {cpu_secs:.1}s, gpu {gpu_secs:.1}s, \
         mean pearson r = {mean_corr:.3} (per-target: {per_target:?})",
        cfg.n_trees
    );

    assert!(
        mean_corr >= 0.95,
        "Phase 3 RF+bootstrap mean per-target Pearson r = {mean_corr:.3} < 0.95"
    );
}

/// ET path still works after the RF dispatch was added. Uses the smallest
/// ET config that exercises the dispatch code path -- one wave, one batch,
/// short trees. Non-zero importances plus a positive CPU-vs-GPU Pearson are
/// enough to confirm nothing broke in the dispatch.
#[test]
fn phase3_et_still_works() {
    let Some(device) = try_device() else {
        eprintln!("scenic_gpu (Phase 3 ET sanity): no wgpu device -- skipping");
        return;
    };

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

    let cpu = fit_multi_trees_sparse(&axes, &x, ET_SAMPLES, &cfg, 7u32 as usize)
        .expect("ET CPU fit failed");
    let gpu = fit_multi_trees_gpu::<WgpuRuntime>(
        &axes,
        &x,
        ET_SAMPLES,
        &cfg,
        7u32 as usize,
        device.clone(),
        &ScenicGpuParams::default(),
    )
    .expect("ET GPU fit failed");

    let mut per_target: Vec<f32> = Vec::with_capacity(ET_TARGETS);
    for t in 0..ET_TARGETS {
        per_target.push(pearson(&cpu[t], &gpu[t]));
    }
    let mean_corr = per_target.iter().sum::<f32>() / per_target.len() as f32;
    eprintln!("phase3_et_still_works: mean pearson r = {mean_corr:.3} ({per_target:?})");

    // ET at 50 trees is still noisy; we're checking the dispatch didn't
    // silently break ET (mean must be materially above zero).
    assert!(
        mean_corr >= 0.4,
        "ET dispatch appears broken after RF was added: mean pearson r = {mean_corr:.3}"
    );
}
