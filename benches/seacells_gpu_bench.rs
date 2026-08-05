//! Benchmarks for the GPU SEACells path, kernel and end-to-end.
//!
//! Three sections. `launch_fw_argmin_b` on a `K²B` of realistic shape and
//! density, reported without a CPU baseline; `launch_fw_columns_a` swept across
//! every workgroup tier against the CPU solve it falls back to, in the same run;
//! then the full fit both ways.
//!
//! Timings are best-of-N with the worst also reported: a single-shot number is a
//! first-call number, since shader compilation and buffer-pool first touch land
//! entirely on run one.
//!
//! Output buffers are zeroed before each timed configuration rather than left
//! `empty()`. A dispatch that busts a device limit is rejected on the cubecl
//! server thread and does no work while reporting success, so every arm checks
//! that the kernel wrote something before any timing is believed. Reusing a
//! buffer across configurations defeats that check, and did.
//!
//! Environment switches, all optional:
//!
//! - `BIXVERSE_BENCH_BIG=1` adds B shapes up to 500k cells
//! - `BIXVERSE_BENCH_A_ONLY=1` runs only the A-column section
//! - `BIXVERSE_BENCH_A_K=10000` restricts it to one archetype count
//! - `BIXVERSE_BENCH_A_WG=512` restricts it to one workgroup width
//!
//! ```bash
//! cargo bench --features gpu,single-cell --bench seacells_gpu_bench
//! ```

use std::time::{Duration, Instant};

use cubecl::future;
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl_utils_rs::prelude::*;
use faer::Mat;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use bixverse_rs::gpu::linalg::sparse_gpu::GpuCompressedSparseData;
use bixverse_rs::gpu::sc_gpu::kernels::seacells_kernels::{
    A_COLUMNS_WG_TIERS, B_ARGMIN_BLOCKS, a_columns_capacity, a_columns_segments,
    b_argmin_workgroup, launch_fw_argmin_b, launch_fw_columns_a,
};
use bixverse_rs::gpu::sc_gpu::seacells_gpu::seacells_fit_gpu;
use bixverse_rs::prelude::{CompressedSparseData2, CompressedSparseFormat};
use bixverse_rs::single_cell::mc_generation::seacells::{
    CpuFwArgminB, FwArgminB, SEACells, SEACellsParams,
};
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

/// One measured configuration.
#[derive(Clone, Copy, Debug)]
struct Shape {
    /// Number of cells
    n: usize,
    /// Number of archetypes
    k: usize,
    /// Fraction of `n * k` that is non-zero in `K²B`
    density: f64,
}

/// Shapes covering the range the kernel is used over.
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
        },
        Shape {
            n: 50_000,
            k: 666,
            density: 0.0308,
        },
        Shape {
            n: 50_000,
            k: 200,
            density: 0.0405,
        },
        // The shape this work exists for. `t1 = A Aᵀ` used to be densified to
        // 381 MB here, and the inner loop read a full k-float row of it per
        // non-zero of the K²B row.
        Shape {
            n: 50_000,
            k: 10_000,
            density: 0.008,
        },
        // Just past the widest tier's slot cap, so this one reports the CPU
        // fallback rather than a timing. Kept as the regression guard: if the
        // cap is ever raised without the launch actually working, this shape
        // trips the "did the kernel do anything" check instead of quietly
        // reporting an implausible speedup.
        Shape {
            n: 20_000,
            k: 12_288,
            density: 0.008,
        },
    ];

    // Large shapes probe where the kernel degrades. Densities here are
    // extrapolated from the measured trend (3.4% at 20k, 3.1% at 50k, 2.3% at
    // 200k), so the absolute times scale with whatever the real density turns
    // out to be.
    if std::env::var("BIXVERSE_BENCH_BIG").is_ok() {
        out.extend([
            Shape {
                n: 200_000,
                k: 1_000,
                density: 0.03,
            },
            Shape {
                n: 500_000,
                k: 1_000,
                density: 0.03,
            },
            Shape {
                n: 500_000,
                k: 2_000,
                density: 0.02,
            },
            Shape {
                n: 500_000,
                k: 4_000,
                density: 0.01,
            },
            Shape {
                n: 500_000,
                k: 6_666,
                density: 0.006,
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
        .unwrap()
    };

    let k2b_parts = make_csr(n, k, density, 1);
    let nnz = k2b_parts.0.len();
    let k2b = upload(k2b_parts);
    let t2 = upload(make_csr(n, k, density * 0.6, 5));
    // B carries roughly `max_fw_iters` atoms per archetype column, so it is far
    // sparser than K²B.
    let b_mat = upload(make_csr(n, k, 25.0 / n as f64, 9));

    // `A Aᵀ` is sparse in the same 1-7% band as the A update's `t1`.
    let t1_csr = make_sparse(k, k, 0.05, 3);
    let t1 = GpuCompressedSparseData::<WgpuRuntime, f32>::from_compressed_sparse_data_2(
        &t1_csr, false, &client,
    )
    .expect("t1 upload failed");
    // Above the widest tier's slot cap the solve runs on the CPU, so there is
    // no kernel here to time.
    let Some(b_wg) = b_argmin_workgroup(k) else {
        println!(
            "n = {:>7}  k = {:>5}  |  no workgroup tier, B argmin runs on CPU",
            n, k
        );
        return;
    };
    let seg_host = a_columns_segments(&t1_csr, b_wg, k.div_ceil(b_wg as usize));
    let t1_seg =
        GpuTensor::<WgpuRuntime, u32>::from_slice(&seg_host, vec![seg_host.len()], &client)
            .unwrap();

    let blocks = B_ARGMIN_BLOCKS.min(n.max(1) as u32) as usize;
    // Zeroed, not `empty()`. `reduce_argmin_blocks` runs even when
    // `fw_argmin_b` was rejected, so uninitialised partials become plausible
    // looking output and the check below passes a kernel that did nothing.
    let part_val = GpuTensor::<WgpuRuntime, f32>::from_slice(
        &vec![0.0f32; blocks * k],
        vec![blocks * k],
        &client,
    )
    .unwrap();
    let part_idx = GpuTensor::<WgpuRuntime, u32>::from_slice(
        &vec![0u32; blocks * k],
        vec![blocks * k],
        &client,
    )
    .unwrap();
    let gap_partial = GpuTensor::<WgpuRuntime, f32>::empty(vec![blocks], &client).unwrap();
    // Zeroed, not `empty()`: a rejected dispatch leaves uninitialised VRAM,
    // which can pass a "did it write anything" check by accident.
    let out_val =
        GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; k], vec![k], &client).unwrap();
    let out_idx =
        GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; k], vec![k], &client).unwrap();

    let mut timings: Vec<Duration> = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let start = Instant::now();
        launch_fw_argmin_b(
            &k2b,
            &t1,
            &t1_seg,
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
    println!(
        "    (k = {}, wg tier {}, {} slots)",
        k,
        b_wg,
        k.div_ceil(b_wg as usize)
    );
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
        + t1_seg.vram_bytes()
        + part_val.vram_bytes()
        + part_idx.vram_bytes();
    println!(
        "n = {:>7}  k = {:>5}  nnz(K2B) = {:>10}  |  GPU best {:>8.2} ms  worst {:>8.2} ms  \
         |  {:>6.1} GFLOP/s  |  {:>7.1} MB VRAM  |  t1 {:>6.1} MB sparse",
        n,
        k,
        nnz,
        best,
        worst,
        flops / (best / 1000.0) / 1e9,
        vram as f64 / (1024.0 * 1024.0),
        (t1.vram_bytes() + t1_seg.vram_bytes()) as f64 / (1024.0 * 1024.0),
    );
}

//////////////////////
// A-column arm     //
//////////////////////

/// One measured A-column configuration.
#[derive(Clone, Copy, Debug)]
struct AShape {
    /// Number of cells
    n: usize,
    /// Number of archetypes
    k: usize,
    /// Fraction of `n * k` that is non-zero in `K²B`
    k2b_density: f64,
    /// Fraction of `k * k` that is non-zero in `t1 = Bᵀ K² B`
    t1_density: f64,
}

/// Shapes for the A-column solve, spanning the current `k <= 2048` ceiling.
///
/// `t1` densities follow the 1-7% range the code records as measured, falling
/// with `k` because an archetype's two-hop neighbourhood does not grow with the
/// archetype count.
///
/// ### Returns
///
/// The shapes to measure.
fn a_shapes() -> Vec<AShape> {
    let mut out = vec![
        AShape {
            n: 50_000,
            k: 666,
            k2b_density: 0.0308,
            t1_density: 0.07,
        },
        AShape {
            n: 50_000,
            k: 2_048,
            k2b_density: 0.025,
            t1_density: 0.04,
        },
        AShape {
            n: 50_000,
            k: 4_096,
            k2b_density: 0.015,
            t1_density: 0.025,
        },
        AShape {
            n: 50_000,
            k: 8_192,
            k2b_density: 0.010,
            t1_density: 0.015,
        },
        // The shape this work exists for, at a tractable cell count: 1M cells at
        // k = 10 000 is the same per-cell kernel, 20x more of it.
        AShape {
            n: 50_000,
            k: 10_000,
            k2b_density: 0.008,
            t1_density: 0.012,
        },
        AShape {
            n: 50_000,
            k: 12_288,
            k2b_density: 0.007,
            t1_density: 0.010,
        },
    ];

    if std::env::var("BIXVERSE_BENCH_BIG").is_ok() {
        out.push(AShape {
            n: 200_000,
            k: 10_000,
            k2b_density: 0.008,
            t1_density: 0.012,
        });
    }
    out
}

/// Build a CSR matrix at the requested density, column indices sorted per row.
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
/// The matrix in CSR.
fn make_sparse(n: usize, k: usize, density: f64, seed: usize) -> CompressedSparseData2<f32> {
    let (values, indices, indptr) = make_csr(n, k, density, seed);
    CompressedSparseData2::new_csr(&values, &indices, &indptr, None, (n, k))
}

/// Build `A_prevᵀ` with a fixed number of atoms per cell, all weights positive
/// and summing to one, so the solve starts from a genuine convex combination.
///
/// ### Params
///
/// * `n` - Cells
/// * `k` - Archetypes
/// * `atoms` - Atoms per cell
///
/// ### Returns
///
/// `A_prevᵀ` as CSR `n × k`.
fn make_a_prev_t(n: usize, k: usize, atoms: usize) -> CompressedSparseData2<f32> {
    let atoms = atoms.min(k);
    let stride = (k / atoms).max(1);

    let mut values = Vec::with_capacity(n * atoms);
    let mut indices = Vec::with_capacity(n * atoms);
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0u32);

    for row in 0..n {
        let phase = (row * 11) % stride;
        let mut col = phase;
        let start = values.len();
        while col < k && values.len() - start < atoms {
            indices.push(col as u32);
            values.push(((row * 13 + col * 7) % 9) as f32 + 1.0);
            col += stride;
        }
        let sum: f32 = values[start..].iter().sum();
        for value in &mut values[start..] {
            *value /= sum;
        }
        indptr.push(values.len() as u32);
    }

    CompressedSparseData2::new_csr(&values, &indices, &indptr, None, (n, k))
}

/// Time the A-column solve on one shape, CPU against GPU.
///
/// Reports the three numbers that decide which wall the kernel hits: achieved
/// arithmetic rate, the `t1` traffic the dense path implies, and the register
/// slots per thread. The dense `t1` row read is `k` floats per Frank-Wolfe
/// iteration per cell, which is the term that grows without bound.
///
/// ### Params
///
/// * `shape` - The configuration
/// * `n_iters` - Frank-Wolfe iterations per column
/// * `device` - Device to run on
fn run_a_shape(shape: AShape, n_iters: usize, device: &WgpuDevice) {
    let client = WgpuRuntime::client(device);
    let AShape {
        n,
        k,
        k2b_density,
        t1_density,
    } = shape;

    let t1 = make_sparse(k, k, t1_density, 3);
    let a_prev_t = make_a_prev_t(n, k, 20);
    let k2_b = make_sparse(n, k, k2b_density, 1);

    let t1_row_nnz = t1.data.len() as f64 / k as f64;
    let cap = a_columns_capacity(&a_prev_t, n_iters);
    let pruning = Some(1e-7f32);

    // CPU baseline through the public seam, so the bench measures what the
    // fallback actually runs.
    let mut cpu_backend = CpuFwArgminB::default();
    let cpu_start = Instant::now();
    let cpu_cols = cpu_backend
        .columns_a(&t1, &a_prev_t, &k2_b, k, n, n_iters, pruning)
        .expect("CPU columns_a failed");
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;
    let cpu_atoms: usize = cpu_cols.iter().map(|c| c.len()).sum();
    drop(cpu_cols);

    // What the dense upload would have cost, kept for comparison: it is
    // quadratic in `k` and was the term that forced the CPU fallback.
    let t1_dense_bytes = k * k * size_of::<f32>();
    let t1_sparse_bytes = t1.data.len() * (size_of::<f32>() + size_of::<u32>());

    let t1_gpu = GpuCompressedSparseData::<WgpuRuntime, f32>::from_compressed_sparse_data_2(
        &t1, false, &client,
    )
    .expect("t1 upload failed");

    let a_prev_gpu = GpuCompressedSparseData::<WgpuRuntime, f32>::from_compressed_sparse_data_2(
        &a_prev_t, false, &client,
    )
    .expect("A_prev upload failed");
    let k2b_gpu = GpuCompressedSparseData::<WgpuRuntime, f32>::from_compressed_sparse_data_2(
        &k2_b, false, &client,
    )
    .expect("K2B upload failed");

    let atom_idx = GpuTensor::<WgpuRuntime, u32>::empty(vec![n * cap], &client).unwrap();
    let threshold =
        GpuTensor::<WgpuRuntime, f32>::from_slice(&[1e-7f32], vec![1], &client).unwrap();

    println!(
        "n = {:>7}  k = {:>5}  cap {:>4}  nnz(t1)/row {:>6.1}  t1 {:>6.1} MB sparse vs \
         {:>7.1} MB dense  |  CPU {:>9.1} ms  ({} atoms)",
        n,
        k,
        cap,
        t1_row_nnz,
        t1_sparse_bytes as f64 / (1024.0 * 1024.0),
        t1_dense_bytes as f64 / (1024.0 * 1024.0),
        cpu_ms,
        cpu_atoms,
    );

    // Sweep the width explicitly rather than taking the tier the selector picks:
    // `slots` and the reduction width move together under the selector, so only
    // a forced sweep separates the register spill from everything else that
    // changes with `k`. `BIXVERSE_BENCH_A_WG=512` narrows it to one width.
    let only_wg: Option<u32> = std::env::var("BIXVERSE_BENCH_A_WG")
        .ok()
        .and_then(|v| v.parse().ok());

    for (wg, max_slots) in A_COLUMNS_WG_TIERS {
        let slots = k.div_ceil(wg as usize);
        // Past the per-tier cap the launch is rejected and the kernel silently
        // does nothing, so the sweep stops where the library would.
        if cap > wg as usize || slots > max_slots || only_wg.is_some_and(|w| w != wg) {
            continue;
        }

        // Zeroed per tier, or the previous tier's output passes the mass check
        // for a launch that did nothing. That is not hypothetical: it hid a
        // silent failure at k = 16384 until the buffers were cleared.
        let atom_cnt =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; n], vec![n], &client).unwrap();
        let atom_val = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![0.0f32; n * cap],
            vec![n * cap],
            &client,
        )
        .unwrap();

        let seg_host = a_columns_segments(&t1, wg, slots);
        let t1_seg =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&seg_host, vec![seg_host.len()], &client)
                .unwrap();

        let mut timings: Vec<Duration> = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let start = Instant::now();
            launch_fw_columns_a(
                &t1_gpu,
                &t1_seg,
                &a_prev_gpu,
                &k2b_gpu,
                &atom_idx,
                &atom_val,
                &atom_cnt,
                &threshold,
                n,
                k,
                n_iters,
                cap as u32,
                pruning,
                None,
                Some(wg),
                &client,
            )
            .expect("launch failed");
            future::block_on(client.sync()).expect("device sync failed");
            timings.push(start.elapsed());
        }

        timings.sort();
        let best = timings[0].as_secs_f64() * 1000.0;
        let worst = timings[REPEATS - 1].as_secs_f64() * 1000.0;

        // A launch that busts a device limit does no work and returns zeros.
        // Every column renormalises to a convex combination, so the atom weights
        // sum to `n`.
        let val_host = atom_val.clone().read(&client).expect("read failed");
        let cnt_host = atom_cnt.clone().read(&client).expect("read failed");
        let mass: f64 = (0..n)
            .map(|cell| {
                let count = cnt_host[cell] as usize;
                val_host[cell * cap..cell * cap + count]
                    .iter()
                    .map(|v| *v as f64)
                    .sum::<f64>()
            })
            .sum();
        assert!(
            mass > 0.5 * n as f64,
            "kernel almost certainly did no work at wg {wg}: atom mass {mass}, expected {n}"
        );

        // What the dense path cost: one k-float row read per Frank-Wolfe iteration per
        // cell, which is the term that grows without bound.
        let t1_traffic = n as f64 * n_iters as f64 * k as f64 * 4.0;
        // Register-array scan work: argmin plus the convex-step update, both
        // `slots` wide, per iteration per thread.
        let flops = n as f64 * n_iters as f64 * k as f64 * 2.0;
        // Shared memory decides how many workgroups stay resident per core.
        let smem = 3 * cap.next_power_of_two() * 4 + 2 * (wg as usize / 32) * 4 + 12;

        println!(
            "    wg {:>5}  slots {:>3}  smem {:>6} B  resident {:>2}  |  best {:>9.1} ms  \
             worst {:>9.1} ms  |  {:>6.2}x CPU  |  {:>6.1} GFLOP/s  {:>6.1} GB/s t1",
            wg,
            slots,
            smem,
            32768 / smem.max(1),
            best,
            worst,
            cpu_ms / best,
            flops / (best / 1000.0) / 1e9,
            t1_traffic / (best / 1000.0) / 1e9,
        );
    }
}

//////////////
// End-to-end   //
//////////////////

/// Clustered synthetic embedding.
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
    // The A-column arm is the one under active work, so it can be run alone
    // rather than behind several minutes of end-to-end fits.
    let a_only = std::env::var("BIXVERSE_BENCH_A_ONLY").is_ok();

    if !a_only {
        println!(
            "\nSEACells B-gradient argmin: GPU kernel vs measured CPU baseline\n\
             (CPU numbers from benches/seacells_bench.rs, pruning 1e-7)\n"
        );
        for shape in shapes() {
            run_shape(shape, &device);
        }
    }

    {
        let limits = GpuLimits::from_client(&WgpuRuntime::client(&device));
        println!(
            "\ndevice: max_units_per_cube {}  max_cube_dim {:?}  max_cube_count {:?}  \
             plane {}..{}  shared {} B  binding limit {} MB",
            limits.max_units_per_cube,
            limits.max_cube_dim,
            limits.max_cube_count,
            limits.plane_size_min,
            limits.plane_size_max,
            limits.max_shared_bytes,
            limits.max_binding_bytes / (1024 * 1024),
        );
    }

    println!(
        "\nSEACells A-column Frank-Wolfe solve: GPU kernel vs the CPU path it falls back to\n\
         (50 Frank-Wolfe iterations, pruning 1e-7; every tier the shape fits is swept, since \
         the narrowest is not always the fastest)\n"
    );
    // Restricts the sweep to one archetype count, so a shape can be measured
    // without the buffers of every earlier shape still sitting in the
    // allocator's pool.
    let only_k: Option<usize> = std::env::var("BIXVERSE_BENCH_A_K")
        .ok()
        .and_then(|v| v.parse().ok());
    for shape in a_shapes() {
        if only_k.is_some_and(|k| k != shape.k) {
            continue;
        }
        run_a_shape(shape, 50, &device);
    }
    if a_only {
        println!();
        return;
    }

    println!("\nEnd to end, full fit, pruning 1e-7, 3 outer iterations\n");
    for (n, k, knn) in [
        (6_800usize, 250usize, 15usize),
        (20_000, 266, 15),
        (50_000, 666, 15),
    ] {
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
