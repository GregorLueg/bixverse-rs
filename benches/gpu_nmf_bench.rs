//! End-to-end benchmark for the GPU HALS NMF path.
//!
//! Two questions, and they need different measurements.
//!
//! **Does the GPU beat the CPU, and where.** Every shape runs the CPU solver and
//! the GPU solver over the same matrix in the same pass, so the comparison never
//! spans machines or build profiles. A single end-to-end number would hide
//! everything that matters, so each GPU run is also split into the three stages
//! the public API can actually separate, with a device sync between them: upload
//! of `V`, scratch allocation, and the solve. The staged and pipelined totals do
//! not add up, which is expected: staged measures isolated cost, pipelined
//! measures a queue.
//!
//! Anything finer than those three stages is not measured here. Apportioning a
//! single solve across its kernels by re-running pieces would mean either
//! instrumenting the shipping solver or duplicating its loop and letting the two
//! drift, and the cubecl profiler already gives exact per-kernel attribution with
//! no code changes. Use it, and read the spread rather than the mean: one kernel
//! name covering two clearly separated timings means two shapes.
//!
//! **Whether the dense products need their own kernel.** They did, and this bench
//! is what established it: cubek asked for 40 KiB of shared memory against a
//! 32 KiB device on the metacell shape and would not launch at all, and on the
//! one shape where it did run it managed 0.1% of peak, putting the whole solve at
//! 0.18x the CPU. Both products now go through
//! [`bixverse_rs::gpu::linalg::skinny_gemm`]. The achieved GFLOP/s and GB/s
//! columns are what keep that honest: a solve at a few percent of both is bound
//! by neither, and that regime has its own fix list.
//!
//! The restart section is the one that matters most for the real workload. A
//! consensus k sweep is `k_range.len() * n_runs` solves, and the GPU path uploads
//! `V` once for all of them while the CPU pays full memory traffic over `V` every
//! time. A single-solve comparison understates that completely.
//!
//! ### Baseline, M1 Max, 60 iterations per solve
//!
//! | shape | cpu | gpu | ratio |
//! |---|---|---|---|
//! | bulk 500 x 20000, k = 10 | 375 ms | 172 ms | 2.18x |
//! | metacell 5000 x 3000, k = 30 | 842 ms | 302 ms | 2.79x |
//! | sc-sparse 50k x 3000, k = 30, 5% | 4.40 s | 512 ms | 8.61x |
//! | sc-sparse 200k x 3000, k = 30, 5% | 29.4 s | 2.03 s | 14.5x |
//! | metacell restarts, 8 runs | 3.59 s | 2.02 s | 1.78x |
//!
//! The sparse arm sits at 56% of device bandwidth, which is where an SpMM with a
//! `k`-wide dense gather per non-zero belongs, so there is little left there. The
//! dense arm is at 2.0 and 4.5% of peak FLOPs against 10.3 and 7.7% of bandwidth,
//! so it is bound by neither and the register tile is the lever if anyone wants
//! more. Restarts gain least because the CPU fans out across cores there while
//! the GPU runs them one after another.
//!
//! Run with: cargo bench --bench gpu_nmf_bench --features gpu
//! Add BIXVERSE_BENCH_BIG=1 for the large single-cell shape.
//! Kernel attribution: CUBECL_DEBUG_OPTION=profile-medium CUBECL_DEBUG_LOG=stdout

#![allow(missing_docs)]

use std::time::{Duration, Instant};

use cubecl::future;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use faer::Mat;

use cubecl_utils_rs::prelude::*;

use bixverse_rs::core::math::sparse::CompressedSparseData2;
use bixverse_rs::gpu::methods_gpu::nmf_gpu::{
    GpuDenseNmfInput, GpuSparseNmfInput, NmfGpuScratch, nmf_hals_gpu, stabilised_nmf_gpu,
};
use bixverse_rs::methods::nmf_hals::dense::DenseInput;
use bixverse_rs::methods::nmf_hals::sparse::SparseInput;
use bixverse_rs::methods::nmf_hals::{HalsOpts, NmfInit, nmf_hals, stabilised_nmf};

////////////
// Consts //
////////////

/// End-to-end repetitions per shape. The reported figure is the best of these,
/// with the worst printed alongside: a single shot is a shader-compilation and
/// buffer-pool-first-touch number, and both paths pay one-off costs that have
/// nothing to do with the steady state.
const REPS: usize = 3;

/// Iterations every solve is pinned to, so the comparison is per-iteration cost
/// rather than a race to a tolerance. Convergence would otherwise let one path
/// stop earlier and look faster for the wrong reason.
const FIXED_ITERS: usize = 60;

/// Objective cadence. Matches the `HalsOpts` default, and it matters here: the
/// check recomputes a Gram and a data product and forces a readback, so it is a
/// real share of the per-iteration cost.
const CHECK_EVERY: usize = 10;

/// M1 Max fp32 peak, in GFLOP/s. Only used to turn a measured rate into a
/// percentage; adjust for other hardware or ignore the column.
const PEAK_GFLOPS: f64 = 10_400.0;

/// M1 Max memory bandwidth, in GB/s.
const PEAK_GBS: f64 = 400.0;

////////////
// Shapes //
////////////

/// One synthetic problem. `m` samples by `n` features at rank `k`.
#[derive(Clone, Copy, Debug)]
struct NmfShape {
    /// Human-readable label used in the output.
    label: &'static str,
    /// Samples, i.e. cells or metacells. The reduction length of `W^T V`.
    m: usize,
    /// Features, i.e. genes. The reduction length of `V H^T`.
    n: usize,
    /// Number of components.
    k: usize,
    /// Non-zero fraction for the sparse arm. `None` runs dense only.
    density: Option<f64>,
}

/// Shapes chosen one per regime, because they stress opposite things.
///
/// `bulk` has few samples and many features, so `W^T V` has a short reduction
/// and `V H^T` a long one. `metacell` is squarer and dense. `sc-sparse` is the
/// shape the GPU path exists for: many cells, a few thousand HVGs, sparse.
const DEFAULT_SHAPES: [NmfShape; 3] = [
    NmfShape {
        label: "bulk",
        m: 500,
        n: 20_000,
        k: 10,
        density: None,
    },
    NmfShape {
        label: "metacell",
        m: 5_000,
        n: 3_000,
        k: 30,
        density: None,
    },
    NmfShape {
        label: "sc-sparse",
        m: 50_000,
        n: 3_000,
        k: 30,
        density: Some(0.05),
    },
];

/// The production single-cell shape, behind `BIXVERSE_BENCH_BIG`. Dense would be
/// 1.2 GB at this size, so it runs sparse only.
const BIG_SHAPES: [NmfShape; 1] = [NmfShape {
    label: "sc-sparse-big",
    m: 200_000,
    n: 3_000,
    k: 30,
    density: Some(0.05),
}];

/////////////
// Helpers //
/////////////

/// Solver options pinned to a fixed iteration count.
///
/// ### Returns
///
/// [`HalsOpts`] with the tolerance set below anything reachable, so every run
/// does exactly [`FIXED_ITERS`] iterations.
fn bench_opts() -> HalsOpts<f32> {
    HalsOpts::<f32> {
        max_iter: FIXED_ITERS,
        tol: 0.0,
        eps: 1e-10,
        check_every: CHECK_EVERY,
        init: NmfInit::Random { seed: 7 },
    }
}

/// Deterministic non-negative dense matrix with a rank-`k` core plus noise.
///
/// The core matters: a matrix with no low-rank structure gives HALS nothing to
/// converge towards, and the per-iteration cost is what is being measured, not
/// the trajectory. The per-entry hash keeps the condition number sane.
///
/// ### Params
///
/// * `m` - Rows
/// * `n` - Columns
/// * `k` - Rank of the planted core
///
/// ### Returns
///
/// A non-negative `m x n` matrix.
fn build_dense(m: usize, n: usize, k: usize) -> Mat<f32> {
    let w = Mat::<f32>::from_fn(m, k, |i, c| {
        (((i * 2_654_435_761usize).wrapping_add(c * 40_503) % 1_009) as f32) / 1_009.0 + 0.01
    });
    let h = Mat::<f32>::from_fn(k, n, |r, j| {
        (((j * 2_246_822_519usize).wrapping_add(r * 97_711) % 1_013) as f32) / 1_013.0 + 0.01
    });
    w * h
}

/// Deterministic non-negative sparse matrix in CSR, samples by features.
///
/// Every column gets its own support pattern rather than a shared one: a handful
/// of distinct column supports puts enough block structure into the matrix to
/// wreck its condition number, which shows up as an initialisation failure that
/// looks like a library bug.
///
/// ### Params
///
/// * `m` - Rows
/// * `n` - Columns
/// * `density` - Target non-zero fraction
///
/// ### Returns
///
/// The matrix in CSR, plus its non-zero count.
fn build_sparse(m: usize, n: usize, density: f64) -> (CompressedSparseData2<f32>, usize) {
    let stride = (1.0 / density).round().max(1.0) as usize;
    let mut values: Vec<f32> = Vec::with_capacity(m * n / stride + m);
    let mut indices: Vec<u32> = Vec::with_capacity(m * n / stride + m);
    let mut indptr: Vec<u32> = Vec::with_capacity(m + 1);
    indptr.push(0);

    for i in 0..m {
        // Row-dependent phase, so no two rows share a support.
        let phase = (i * 7_919) % stride;
        let mut j = phase;
        while j < n {
            let h = (i * 2_654_435_761usize).wrapping_add(j * 2_246_822_519usize) % 1_021;
            values.push((h as f32) / 1_021.0 + 0.01);
            indices.push(j as u32);
            j += stride;
        }
        indptr.push(values.len() as u32);
    }

    let nnz = values.len();
    (
        CompressedSparseData2::<f32, f32>::new_csr(&values, &indices, &indptr, None, (m, n)),
        nnz,
    )
}

/// Block until every queued kernel has retired.
///
/// ### Params
///
/// * `client` - CubeCL compute client
fn sync<R: Runtime>(client: &ComputeClient<R>) {
    future::block_on(client.sync()).expect("device sync failed");
}

/// Time a closure, returning its value and how long it took.
///
/// ### Params
///
/// * `f` - The closure to time
///
/// ### Returns
///
/// The closure's value and the elapsed duration.
fn timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let out = f();
    (out, start.elapsed())
}

/// Best and worst of `REPS` runs of a closure.
///
/// ### Params
///
/// * `f` - The closure to repeat
///
/// ### Returns
///
/// `(best, worst)` durations. Everything above the best is shader compilation,
/// buffer-pool first touch and host allocator faulting, so both are worth
/// reporting.
fn best_of(mut f: impl FnMut() -> Duration) -> (Duration, Duration) {
    let mut best = Duration::MAX;
    let mut worst = Duration::ZERO;
    for _ in 0..REPS {
        let d = f();
        best = best.min(d);
        worst = worst.max(d);
    }
    (best, worst)
}

/// Guard that a solve did real work.
///
/// A rejected dispatch leaves its output untouched and reports success, so an
/// implausibly fast GPU result is a silent failure until a checksum says
/// otherwise. A converging HALS run has a finite positive loss below `||V||^2`.
///
/// ### Params
///
/// * `label` - Shape label for the message
/// * `loss` - Reported reconstruction error
/// * `sq_frob` - `||V||_F^2`
fn assert_real_work(label: &str, loss: f32, sq_frob: f32) {
    assert!(
        loss.is_finite() && loss > 0.0,
        "{label}: loss {loss} is not a real reconstruction error, the GPU probably did no work"
    );
    assert!(
        loss < sq_frob,
        "{label}: loss {loss} is above ||V||^2 = {sq_frob}, the factors are worse than zero"
    );
}

///////////////////////
// Rates and staging //
///////////////////////

/// Print achieved rates against device peak.
///
/// Three numbers taken together say which wall a kernel hits, and no two of them
/// say it alone: a solve at a few percent of both compute and bandwidth is bound
/// by neither, and the fix list for that case does not overlap with the fix list
/// for either of the others.
///
/// These are whole-solve rates against the per-iteration budget, not per-kernel.
/// Per-kernel attribution comes from the cubecl profiler; see the module doc.
///
/// ### Params
///
/// * `name` - Label
/// * `flops` - Floating-point operations per iteration
/// * `bytes` - Bytes moved per iteration
/// * `elapsed` - Time for `iters` iterations
/// * `iters` - Number of iterations
fn report_rates(name: &str, flops: f64, bytes: f64, elapsed: Duration, iters: usize) {
    let per = elapsed.as_secs_f64() / iters as f64;
    let gflops = flops / per / 1e9;
    let gbs = bytes / per / 1e9;
    println!(
        "    {name:<14} {:>8.2} ms/iter  {:>8.1} GFLOP/s ({:>4.1}% peak)  {:>7.1} GB/s ({:>4.1}%)",
        per * 1e3,
        gflops,
        100.0 * gflops / PEAK_GFLOPS,
        gbs,
        100.0 * gbs / PEAK_GBS
    );
}

/// Upload, scratch allocation and solve, each with a device sync.
///
/// These three are separable through the public API, so they are measured rather
/// than apportioned. Anything finer would mean either instrumenting the shipping
/// solver or re-implementing its loop here and letting the two drift; the cubecl
/// profiler already gives exact per-kernel attribution at neither cost.
///
/// ### Params
///
/// * `label` - Shape label
/// * `upload` - Upload of `V`
/// * `scratch` - Scratch allocation
/// * `solve` - The solve itself
fn report_stages(label: &str, upload: Duration, scratch: Duration, solve: Duration) {
    let total = (upload + scratch + solve).as_secs_f64();
    let row = |name: &str, d: Duration| {
        println!(
            "    {name:<14} {:>9.2?}  {:>5.1}%",
            d,
            100.0 * d.as_secs_f64() / total
        );
    };
    println!("  staged, syncs between ({label}):");
    row("upload V", upload);
    row("scratch", scratch);
    row("solve", solve);
}

///////////////
// Dense arm //
///////////////

/// Run the dense arm of one shape: CPU against GPU, end to end and staged.
///
/// ### Params
///
/// * `shape` - The problem to run
/// * `device` - CubeCL device
fn run_dense<R: Runtime>(shape: NmfShape, device: &R::Device) {
    let NmfShape { label, m, n, k, .. } = shape;
    println!("-- {label} dense: {m} x {n}, k = {k}, {FIXED_ITERS} iters --");

    let v = build_dense(m, n, k);
    let opts = bench_opts();
    let client = R::client(device);

    // CPU baseline, in the same pass so the comparison never spans machines.
    let cpu_in = DenseInput::new(v.as_ref()).expect("CPU input");
    let sq_frob = bixverse_rs::methods::nmf_hals::NmfInput::sq_frob(&cpu_in);
    let mut cpu_loss = 0f32;
    let (cpu_best, cpu_worst) = best_of(|| {
        let (res, d) = timed(|| nmf_hals(&cpu_in, k, &opts, 0).expect("CPU solve"));
        cpu_loss = res.final_loss;
        d
    });
    assert_real_work(label, cpu_loss, sq_frob);

    // GPU, end to end with the queue left to pipeline.
    let mut gpu_loss = 0f32;
    let (gpu_best, gpu_worst) = best_of(|| {
        let (d, elapsed) = timed(|| {
            let input = GpuDenseNmfInput::<R>::new(v.as_ref(), &client).expect("GPU input");
            let scratch = NmfGpuScratch::<R>::new(m, n, k, opts.eps, &client).expect("GPU scratch");
            let res = nmf_hals_gpu(&input, k, &opts, &scratch, &client, 0).expect("GPU solve");
            sync(&client);
            res.final_loss
        });
        gpu_loss = d;
        elapsed
    });
    assert_real_work(label, gpu_loss, sq_frob);

    let rel_cpu = cpu_loss / sq_frob;
    let rel_gpu = gpu_loss / sq_frob;
    println!(
        "  cpu {:>9.2?} (worst {:>9.2?})   gpu {:>9.2?} (worst {:>9.2?})   {:.2}x",
        cpu_best,
        cpu_worst,
        gpu_best,
        gpu_worst,
        cpu_best.as_secs_f64() / gpu_best.as_secs_f64()
    );
    println!("  relative loss: cpu {rel_cpu:.6}  gpu {rel_gpu:.6}  (they solve the same problem)");

    run_dense_staged::<R>(shape, v.as_ref(), &client);
}

/// The staged breakdown for the dense arm.
/// The staged breakdown for the dense arm, plus the per-iteration rate budget.
///
/// ### Params
///
/// * `shape` - The problem to run
/// * `v` - The matrix, borrowed
/// * `client` - CubeCL compute client
fn run_dense_staged<R: Runtime>(shape: NmfShape, v: faer::MatRef<f32>, client: &ComputeClient<R>) {
    let NmfShape { label, m, n, k, .. } = shape;
    let opts = bench_opts();

    let (input, upload) = timed(|| {
        let i = GpuDenseNmfInput::<R>::new(v, client).expect("GPU input");
        sync(client);
        i
    });
    let (scratch, alloc) = timed(|| {
        let s = NmfGpuScratch::<R>::new(m, n, k, opts.eps, client).expect("GPU scratch");
        sync(client);
        s
    });
    let (_, solve) = timed(|| {
        nmf_hals_gpu(&input, k, &opts, &scratch, client, 0).expect("GPU solve");
        sync(client);
    });

    report_stages(label, upload, alloc, solve);

    // Both dense products are 2*m*n*k, so the per-iteration budget is twice
    // that, plus the objective's extra product every CHECK_EVERY iterations.
    let per_product = 2.0 * m as f64 * n as f64 * k as f64;
    let extra = 1.0 / CHECK_EVERY as f64;
    let flops = (2.0 + extra) * per_product;
    // The dominant traffic is reading V once per product. The Grams, the sweeps
    // and the normalisation are all O((m + n) * k) and round to nothing beside it.
    let bytes = (2.0 + extra) * m as f64 * n as f64 * 4.0;
    println!("  per-iteration budget ({label}), dense products dominate:");
    report_rates("solve", flops, bytes, solve, FIXED_ITERS);
    println!(
        "    scratch on device: {:.1} MB",
        scratch.vram_bytes() as f64 / 1e6
    );
}

////////////////
// Sparse arm //
////////////////

/// Run the sparse arm of one shape: CPU against GPU.
///
/// ### Params
///
/// * `shape` - The problem to run
/// * `device` - CubeCL device
fn run_sparse<R: Runtime>(shape: NmfShape, device: &R::Device) {
    let NmfShape {
        label,
        m,
        n,
        k,
        density,
    } = shape;
    let Some(density) = density else {
        return;
    };

    let (csr, nnz) = build_sparse(m, n, density);
    println!(
        "-- {label} sparse: {m} x {n}, k = {k}, nnz = {nnz} ({:.1}% dense), {FIXED_ITERS} iters --",
        100.0 * nnz as f64 / (m as f64 * n as f64)
    );

    let opts = bench_opts();
    let client = R::client(device);

    let cpu_in = SparseInput::<f32, f32>::from_primary(&csr).expect("CPU sparse input");
    let sq_frob = bixverse_rs::methods::nmf_hals::NmfInput::sq_frob(&cpu_in);
    let mut cpu_loss = 0f32;
    let (cpu_best, cpu_worst) = best_of(|| {
        let (res, d) = timed(|| nmf_hals(&cpu_in, k, &opts, 0).expect("CPU solve"));
        cpu_loss = res.final_loss;
        d
    });
    assert_real_work(label, cpu_loss, sq_frob);

    let mut gpu_loss = 0f32;
    let mut upload = Duration::ZERO;
    let (gpu_best, gpu_worst) = best_of(|| {
        let host = SparseInput::<f32, f32>::from_primary(&csr).expect("host sparse input");
        let (d, elapsed) = timed(|| {
            let (input, up) = timed(|| GpuSparseNmfInput::<R>::new(host, &client).expect("upload"));
            sync(&client);
            upload = up;
            let scratch = NmfGpuScratch::<R>::new(m, n, k, opts.eps, &client).expect("GPU scratch");
            let res = nmf_hals_gpu(&input, k, &opts, &scratch, &client, 0).expect("GPU solve");
            sync(&client);
            res.final_loss
        });
        gpu_loss = d;
        elapsed
    });
    assert_real_work(label, gpu_loss, sq_frob);

    println!(
        "  cpu {:>9.2?} (worst {:>9.2?})   gpu {:>9.2?} (worst {:>9.2?})   {:.2}x",
        cpu_best,
        cpu_worst,
        gpu_best,
        gpu_worst,
        cpu_best.as_secs_f64() / gpu_best.as_secs_f64()
    );
    println!(
        "  relative loss: cpu {:.6}  gpu {:.6}",
        cpu_loss / sq_frob,
        gpu_loss / sq_frob
    );
    println!(
        "  of which upload {:>9.2?} ({:.1}% of the GPU run, paid once per sweep not per solve)",
        upload,
        100.0 * upload.as_secs_f64() / gpu_best.as_secs_f64()
    );

    // Both SpMM kernels move the whole non-zero set plus a k-wide dense gather
    // per non-zero, so the arithmetic intensity is fixed and low. This is the
    // number that says whether the sparse arm is bandwidth bound.
    let flops = 2.0 * nnz as f64 * k as f64;
    let bytes = nnz as f64 * (4.0 + 4.0 + 4.0 * k as f64);
    println!("  per-iteration SpMM budget (both directions):");
    report_rates("spmm x2", 2.0 * flops, 2.0 * bytes, gpu_best, FIXED_ITERS);
}

//////////////////////
// Restart section  //
//////////////////////

/// Compare restarts, where the GPU path amortises one upload of `V`.
///
/// This is the comparison that matters for a consensus k sweep: the CPU repeats
/// its full pass over `V` for every solve, and the GPU does not. A single-solve
/// figure understates the real workload.
///
/// ### Params
///
/// * `shape` - The problem to run
/// * `n_runs` - Restarts to run
/// * `device` - CubeCL device
fn run_restarts<R: Runtime>(shape: NmfShape, n_runs: usize, device: &R::Device) {
    let NmfShape { label, m, n, k, .. } = shape;
    println!("-- {label} restarts: {n_runs} runs at k = {k}, {FIXED_ITERS} iters each --");

    let v = build_dense(m, n, k);
    let opts = bench_opts();
    let client = R::client(device);

    let cpu_in = DenseInput::new(v.as_ref()).expect("CPU input");
    let (_, cpu) = timed(|| stabilised_nmf(&cpu_in, k, n_runs, 0, &opts, 0).expect("CPU restarts"));

    let (_, gpu) = timed(|| {
        let input = GpuDenseNmfInput::<R>::new(v.as_ref(), &client).expect("GPU input");
        let scratch = NmfGpuScratch::<R>::new(m, n, k, opts.eps, &client).expect("GPU scratch");
        let res = stabilised_nmf_gpu(&input, k, n_runs, 0, &opts, &scratch, &client, 0)
            .expect("GPU restarts");
        sync(&client);
        res
    });

    println!(
        "  cpu {:>9.2?}   gpu {:>9.2?}   {:.2}x   (cpu fans out over rayon, gpu is serial)",
        cpu,
        gpu,
        cpu.as_secs_f64() / gpu.as_secs_f64()
    );
}

//////////
// Main //
//////////

fn main() {
    let device: <WgpuRuntime as Runtime>::Device = Default::default();
    let limits = GpuLimits::from_client(&WgpuRuntime::client(&device));

    println!("====== GPU HALS NMF bench ======\n");
    println!(
        "device: shared memory {} B, per-binding {} B, plane {}..{}\n",
        limits.max_shared_bytes,
        limits.max_binding_bytes,
        limits.plane_size_min,
        limits.plane_size_max
    );

    for shape in DEFAULT_SHAPES {
        if shape.density.is_none() {
            run_dense::<WgpuRuntime>(shape, &device);
        } else {
            run_sparse::<WgpuRuntime>(shape, &device);
        }
        println!();
    }

    run_restarts::<WgpuRuntime>(DEFAULT_SHAPES[1], 8, &device);
    println!();

    if std::env::var("BIXVERSE_BENCH_BIG").is_ok() {
        for shape in BIG_SHAPES {
            run_sparse::<WgpuRuntime>(shape, &device);
            println!();
        }
    } else {
        println!("Set BIXVERSE_BENCH_BIG=1 to also run the 200k-cell sparse shape.");
    }
}
