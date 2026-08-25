//! Where the pairwise gene correlation wall clock actually goes.
//!
//! Three separable questions, one cell each, because two independent changes
//! landed together and lumping them would let either take credit for the other:
//!
//! * `kernels` - the 128-bit reduction and dot arms with one accumulator
//!   against the same arms with `UNROLL` of them. Single threaded and in cache,
//!   which is the only place the difference is visible: under rayon the memory
//!   system saturates long before the ALUs do.
//! * `dot` - the scalar `map(|(x, y)| x * y).sum::<f32>()` the correlation loop
//!   used against the vectorised dot. LLVM cannot reassociate `f32` addition
//!   without fast-math, so the scalar form is a serial FADD chain and this is
//!   the gap that opens.
//! * `endtoend` - the whole function, dense algorithm against sparse. This is
//!   the number that matters. It runs under rayon on realistic sparsity, both
//!   Pearson and Spearman.
//!
//! The dense baseline in `endtoend` is the algorithm as it was, but calling the
//! current reduction kernels. That is deliberate: it isolates the algorithmic
//! change from the unrolling, which `kernels` already measured on its own.
//!
//! Run with:
//! ```
//! cargo bench --features single-cell --bench pairwise_cor_bench
//! ```
//!
//! `PAIRWISE_BENCH_ONLY=dot,endtoend` runs a comma-separated subset of the
//! cells. `PAIRWISE_BENCH_CELLS` overrides the end-to-end cell counts.

#![cfg(feature = "single-cell")]

use std::hint::black_box;
use std::time::{Duration, Instant};

use bixverse_rs::core::math::vector_helpers::rank_vector;
use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::mc_analysis::metrics::pairwise_gene_correlations_in_memory;
use bixverse_rs::utils::simd::{sum_simd_f32, sum_squared_dev_simd_f32, sum_widen_simd_f32};
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use wide::f32x4;

/// Repeats per timing, chosen so even the smallest case runs long enough to
/// clear timer noise without the largest one dominating the run.
const MIN_ITERS: usize = 3;

/// Target time per measured cell. Iterations are added until this is reached.
const TARGET: Duration = Duration::from_millis(150);

///////////////////////
// Baseline kernels //
///////////////////////

/// The 128-bit sum as it was before the unroll: one accumulator, so one FADD
/// per four elements against roughly three cycles of latency.
#[allow(
    clippy::needless_range_loop,
    reason = "verbatim copy of the pre-unroll kernel"
)]
fn sum_sse_f32_single_acc(a: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = f32x4::ZERO;

    // SAFETY: `chunks * 4 <= len`, so every load stays in bounds.
    unsafe {
        let a_ptr = a.as_ptr();
        for i in 0..chunks {
            let va = f32x4::from(*(a_ptr.add(i * 4) as *const [f32; 4]));
            acc += va;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        sum += a[i];
    }
    sum
}

/// The 128-bit sum of squared deviations as it was, one accumulator.
#[allow(
    clippy::needless_range_loop,
    reason = "verbatim copy of the pre-unroll kernel"
)]
fn sum_squared_dev_sse_f32_single_acc(a: &[f32], mean: f32) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = f32x4::ZERO;
    let mean_vec = f32x4::splat(mean);

    // SAFETY: as above.
    unsafe {
        let a_ptr = a.as_ptr();
        for i in 0..chunks {
            let va = f32x4::from(*(a_ptr.add(i * 4) as *const [f32; 4]));
            let diff = va - mean_vec;
            acc += diff * diff;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        let diff = a[i] - mean;
        sum += diff * diff;
    }
    sum
}

/// The 128-bit dot as it was, one accumulator.
fn dot_sse_f32_single_acc(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = f32x4::ZERO;

    // SAFETY: as above, and `b` is asserted to be the same length.
    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        for i in 0..chunks {
            let va = f32x4::from(*(a_ptr.add(i * 4) as *const [f32; 4]));
            let vb = f32x4::from(*(b_ptr.add(i * 4) as *const [f32; 4]));
            acc += va * vb;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        sum += a[i] * b[i];
    }
    sum
}

/// The correlation inner loop as it was written: a serial scalar chain.
fn dot_scalar_iter(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

//////////////
// Fixtures //
//////////////

/// Sparse log1p-like gene columns, dense form.
fn dense_columns(n_genes: usize, n_cells: usize, density: f64, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_genes)
        .map(|_| {
            (0..n_cells)
                .map(|_| {
                    if rng.random::<f64>() < density {
                        (rng.random::<f32>() * 4.0) + 0.05
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

fn csc_from_columns(columns: &[Vec<f32>]) -> CompressedSparseData2<f32, f32> {
    let n_cells = columns[0].len();
    let mut data: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut indptr: Vec<u32> = vec![0];
    for col in columns {
        for (i, &v) in col.iter().enumerate() {
            if v != 0.0 {
                data.push(v);
                indices.push(i as u32);
            }
        }
        indptr.push(data.len() as u32);
    }
    let data_2 = data.clone();
    CompressedSparseData2::new_csc(
        &data,
        &indices,
        &indptr,
        Some(&data_2),
        (n_cells, columns.len()),
    )
}

/////////////////////
// Dense baseline //
/////////////////////

/// The correlation algorithm as it was: densify every unique gene, standardise
/// it, then one scalar dot per pair.
///
/// Kept here rather than in the library so the comparison is against a fixed
/// target. It calls the current reduction kernels on purpose, so what this
/// measures against the sparse version is the algorithm and nothing else.
fn pairwise_dense_baseline(
    matrix: &CompressedSparseData2<f32, f32>,
    pairs: &[(usize, usize)],
    spearman: bool,
) -> Vec<f32> {
    let n_cells = matrix.shape.0;
    let data_norm = matrix.data_2.as_ref().unwrap();

    let mut unique: Vec<usize> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    unique.sort_unstable();
    unique.dedup();

    let standardised: Vec<Vec<f32>> = unique
        .par_iter()
        .map(|&g| {
            let lo = matrix.indptr[g] as usize;
            let hi = matrix.indptr[g + 1] as usize;
            let mut dense = vec![0_f32; n_cells];
            for p in lo..hi {
                dense[matrix.indices[p] as usize] = data_norm[p];
            }
            let dense = if spearman { rank_vector(&dense) } else { dense };
            let mean = sum_simd_f32(&dense) / n_cells as f32;
            let var = sum_squared_dev_simd_f32(&dense, mean) / (n_cells as f32 - 1.0);
            let std = var.sqrt();
            if std < 1e-8 {
                vec![0_f32; n_cells]
            } else {
                dense.iter().map(|&x| (x - mean) / std).collect()
            }
        })
        .collect();

    let denom = n_cells as f32 - 1.0;
    pairs
        .par_iter()
        .map(|&(g1, g2)| {
            let a = &standardised[unique.binary_search(&g1).unwrap()];
            let b = &standardised[unique.binary_search(&g2).unwrap()];
            (dot_scalar_iter(a, b) / denom).clamp(-1.0, 1.0)
        })
        .collect()
}

//////////////
// Harness //
//////////////

/// Time `f` until [`TARGET`] is reached, return nanoseconds per iteration.
fn time<T>(mut f: impl FnMut() -> T) -> f64 {
    // One untimed pass so caches and any lazy dispatch are warm.
    black_box(f());

    let mut iters = 0usize;
    let start = Instant::now();
    loop {
        black_box(f());
        iters += 1;
        if iters >= MIN_ITERS && start.elapsed() >= TARGET {
            break;
        }
    }
    start.elapsed().as_nanos() as f64 / iters as f64
}

fn enabled(cell: &str) -> bool {
    match std::env::var("PAIRWISE_BENCH_ONLY") {
        Ok(only) => only.split(',').any(|c| c.trim() == cell),
        Err(_) => true,
    }
}

fn speedup(base: f64, new: f64) -> String {
    format!("{:.2}x", base / new)
}

//////////////
// The runs //
//////////////

fn bench_kernels() {
    println!("\n=== kernels: one accumulator vs UNROLL, single threaded, in cache ===");
    println!(
        "{:>8}  {:>12}  {:>12}  {:>8}  {:>12}  {:>12}  {:>8}",
        "len", "sum 1acc", "sum unroll", "gain", "ssd 1acc", "ssd unroll", "gain"
    );

    let mut rng = StdRng::seed_from_u64(7);
    for &len in &[256_usize, 1024, 8192, 65_536, 262_144] {
        let v: Vec<f32> = (0..len).map(|_| rng.random::<f32>()).collect();
        let mean = 0.5_f32;

        let a = time(|| sum_sse_f32_single_acc(black_box(&v)));
        let b = time(|| sum_simd_f32(black_box(&v)));
        let c = time(|| sum_squared_dev_sse_f32_single_acc(black_box(&v), mean));
        let d = time(|| sum_squared_dev_simd_f32(black_box(&v), mean));

        println!(
            "{:>8}  {:>10.0}ns  {:>10.0}ns  {:>8}  {:>10.0}ns  {:>10.0}ns  {:>8}",
            len,
            a,
            b,
            speedup(a, b),
            c,
            d,
            speedup(c, d)
        );
    }

    println!("\n--- widening arms, which the correlation path now uses ---");
    println!(
        "{:>8}  {:>12}  {:>12}  {:>8}",
        "len", "sum f32", "sum widen", "ratio"
    );
    for &len in &[1024_usize, 65_536, 262_144] {
        let v: Vec<f32> = (0..len).map(|_| rng.random::<f32>()).collect();
        let a = time(|| sum_simd_f32(black_box(&v)));
        let b = time(|| sum_widen_simd_f32(black_box(&v)));
        println!(
            "{:>8}  {:>10.0}ns  {:>10.0}ns  {:>8}",
            len,
            a,
            b,
            speedup(a, b)
        );
    }
}

fn bench_dot() {
    println!("\n=== dot: the correlation inner loop, single threaded ===");
    println!(
        "{:>8}  {:>12}  {:>12}  {:>8}  {:>12}  {:>8}",
        "len", "scalar iter", "wide 1acc", "gain", "wide unroll", "gain"
    );

    let mut rng = StdRng::seed_from_u64(11);
    for &len in &[256_usize, 1024, 8192, 65_536, 262_144, 1_048_576] {
        let a: Vec<f32> = (0..len).map(|_| rng.random::<f32>() - 0.5).collect();
        let b: Vec<f32> = (0..len).map(|_| rng.random::<f32>() - 0.5).collect();

        let s = time(|| dot_scalar_iter(black_box(&a), black_box(&b)));
        let one = time(|| dot_sse_f32_single_acc(black_box(&a), black_box(&b)));
        let unr = time(|| {
            let x: f32 = f32::bxv_dot_simd(black_box(&a), black_box(&b));
            x
        });

        println!(
            "{:>8}  {:>10.0}ns  {:>10.0}ns  {:>8}  {:>10.0}ns  {:>8}",
            len,
            s,
            one,
            speedup(s, one),
            unr,
            speedup(s, unr)
        );
    }
}

fn bench_endtoend() {
    println!("\n=== endtoend: dense algorithm vs sparse, under rayon ===");

    let cells: Vec<usize> = match std::env::var("PAIRWISE_BENCH_CELLS") {
        Ok(v) => v.split(',').filter_map(|c| c.trim().parse().ok()).collect(),
        Err(_) => vec![2_000, 20_000, 200_000],
    };

    let n_genes = 40;
    let n_pairs = 200;

    println!(
        "{:>8}  {:>8}  {:>9}  {:>12}  {:>12}  {:>8}  {:>10}",
        "cells", "density", "method", "dense", "sparse", "gain", "max |diff|"
    );

    for &n_cells in &cells {
        for &density in &[0.05_f64, 0.20] {
            let columns = dense_columns(n_genes, n_cells, density, 42);
            let matrix = csc_from_columns(&columns);

            // Distinct genes only. A self-pair correlates a column with
            // itself, where the dense path sums n positive terms and drifts far
            // more than it does on a real pair, which would put a number in the
            // diff column that says nothing about the general case.
            let mut rng = StdRng::seed_from_u64(3);
            let pairs: Vec<(usize, usize)> = (0..n_pairs)
                .map(|_| {
                    let a = rng.random_range(0..n_genes);
                    let mut b = rng.random_range(0..n_genes - 1);
                    if b >= a {
                        b += 1;
                    }
                    (a, b)
                })
                .collect();
            let g1: Vec<usize> = pairs.iter().map(|p| p.0).collect();
            let g2: Vec<usize> = pairs.iter().map(|p| p.1).collect();

            for spearman in [false, true] {
                let dense = time(|| pairwise_dense_baseline(&matrix, &pairs, spearman));
                let sparse = time(|| {
                    pairwise_gene_correlations_in_memory(&matrix, &g1, &g2, spearman, 0).unwrap()
                });

                // Not an assertion. This is the dense path's f32 accumulation
                // error, which the sparse form does not carry: against an f64
                // reference the sparse side sits around 2e-10 while the dense
                // side is at 1e-5, so effectively all of this column is the
                // dense path drifting.
                let want = pairwise_dense_baseline(&matrix, &pairs, spearman);
                let got =
                    pairwise_gene_correlations_in_memory(&matrix, &g1, &g2, spearman, 0).unwrap();
                let max_diff = want
                    .iter()
                    .zip(got.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0_f32, f32::max);

                println!(
                    "{:>8}  {:>8.2}  {:>9}  {:>10.2}ms  {:>10.2}ms  {:>8}  {:>10.2e}",
                    n_cells,
                    density,
                    if spearman { "spearman" } else { "pearson" },
                    dense / 1e6,
                    sparse / 1e6,
                    speedup(dense, sparse),
                    max_diff
                );
            }
        }
    }
}

fn main() {
    println!(
        "threads: {}, target per cell: {:?}",
        rayon::current_num_threads(),
        TARGET
    );

    if enabled("kernels") {
        bench_kernels();
    }
    if enabled("dot") {
        bench_dot();
    }
    if enabled("endtoend") {
        bench_endtoend();
    }
}
