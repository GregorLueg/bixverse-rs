//! HALS NMF: CPU against MLX, plus the cubecl path when `gpu` is on.
//!
//! Same shapes, sparse generator and fixed iteration count as
//! `gpu_nmf_bench`, so the numbers line up. The default run is the sparse
//! single-cell shape; `BIXVERSE_BENCH_BIG=1` adds the 200k-cell sparse shape
//! and `BIXVERSE_BENCH_DENSE=1` adds the dense bulk and metacell shapes.
//!
//! The MLX run is split into its three separable stages: upload of `V` (the
//! host-to-device copy, after host-side validation for sparse input), the solve
//! (blocks on its last objective, so the device is idle when it returns) and
//! the readback of the factors. The cubecl solve reads its factors back
//! internally, so its readback is inside the solve figure.
//!
//! Run with:
//! ```text
//! cargo bench --features mlx --bench mlx_nmf_bench
//! cargo bench --features mlx,gpu --bench mlx_nmf_bench
//! ```
//!
//! Needs `libmlxc` (see `build.rs`); without it the `mlx` module does not
//! exist and this does not compile.

#![cfg_attr(not(mlx_available), no_main)]
#![cfg(mlx_available)]

use std::time::{Duration, Instant};

use faer::Mat;

use bixverse_methods::methods::nmf_hals::dense::DenseInput;
use bixverse_methods::methods::nmf_hals::sparse::SparseInput;
use bixverse_methods::methods::nmf_hals::{HalsOpts, NmfInit, NmfInput, nmf_hals};
use bixverse_mlx::methods::nmf::{
    MlxDenseNmfInput, MlxNmfData, MlxSparseNmfInput, nmf_hals_mlx_device,
};
use bixverse_mlx::prelude::*;

///////////////
// Constants //
///////////////

/// Repetitions per arm; the best is reported with the worst alongside.
const REPS: usize = 3;

/// Iterations every solve is pinned to, so the comparison is per-iteration
/// cost rather than a race to a tolerance.
const FIXED_ITERS: usize = 60;

/// Objective cadence, the `HalsOpts` default.
const CHECK_EVERY: usize = 10;

/// Seed for the random factor initialisation.
const SEED: u64 = 7;

////////////
// Shapes //
////////////

/// One synthetic problem, `m` samples by `n` features at rank `k`.
#[derive(Clone, Copy, Debug)]
struct NmfShape {
    /// Label used in the output
    label: &'static str,
    /// Samples
    m: usize,
    /// Features
    n: usize,
    /// Components
    k: usize,
    /// Non-zero fraction; `None` runs dense
    density: Option<f64>,
}

/// The sparse shape of `gpu_nmf_bench`, run by default.
const SPARSE_SHAPES: [NmfShape; 1] = [NmfShape {
    label: "sc-sparse",
    m: 50_000,
    n: 3_000,
    k: 30,
    density: Some(0.05),
}];

/// The large sparse shape of `gpu_nmf_bench`, behind `BIXVERSE_BENCH_BIG`.
const BIG_SHAPES: [NmfShape; 1] = [NmfShape {
    label: "sc-sparse-big",
    m: 200_000,
    n: 3_000,
    k: 30,
    density: Some(0.05),
}];

/// The dense shapes of `gpu_nmf_bench`, behind `BIXVERSE_BENCH_DENSE`.
const DENSE_SHAPES: [NmfShape; 2] = [
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
];

/////////////
// Helpers //
/////////////

/// Solver options pinned to [`FIXED_ITERS`].
///
/// ### Returns
///
/// [`HalsOpts`] with a zero tolerance and random initialisation.
fn bench_opts() -> HalsOpts<f32> {
    HalsOpts::<f32> {
        max_iter: FIXED_ITERS,
        tol: 0.0,
        eps: 1e-10,
        check_every: CHECK_EVERY,
        init: NmfInit::Random { seed: SEED },
    }
}

/// Deterministic non-negative dense matrix with a rank-`k` core, identical to
/// the one in `gpu_nmf_bench`.
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

/// Deterministic non-negative sparse matrix in CSR, identical to the one in
/// `gpu_nmf_bench`. Every row gets its own support phase.
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

/// Time a closure.
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

/// Best and worst of [`REPS`] runs.
///
/// ### Params
///
/// * `f` - The closure to repeat, returning its own duration
///
/// ### Returns
///
/// `(best, worst)`.
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

/// Guard that a solve did real work: a finite positive loss below `||V||^2`.
///
/// ### Params
///
/// * `label` - Arm label for the message
/// * `loss` - Reported reconstruction error
/// * `sq_frob` - `||V||_F^2`
fn assert_real_work(label: &str, loss: f32, sq_frob: f32) {
    assert!(
        loss.is_finite() && loss > 0.0 && loss < sq_frob,
        "{label}: loss {loss} is not a real reconstruction error (||V||^2 = {sq_frob})"
    );
}

//////////
// Arms //
//////////

/// CPU solve.
///
/// ### Params
///
/// * `input` - CPU input over `V`, dense or sparse
/// * `k` - Components
fn bench_cpu<In: NmfInput<f32> + Sync>(input: &In, k: usize) {
    let opts = bench_opts();
    let sq_frob = input.sq_frob();
    let mut loss = 0f32;
    let (best, worst) = best_of(|| {
        let (res, d) = timed(|| nmf_hals(input, k, &opts, 0).expect("CPU solve"));
        loss = res.final_loss;
        d
    });
    assert_real_work("cpu", loss, sq_frob);
    println!(
        "  cpu    {best:>9.2?} (worst {worst:>9.2?})  rel loss {:.6}",
        loss / sq_frob
    );
}

/// MLX solve, staged.
///
/// ### Params
///
/// * `upload` - Builds the input, returning it and the time spent on the
///   device copy alone
/// * `k` - Components
fn bench_mlx<In: MlxNmfData>(mut upload: impl FnMut() -> (In, Duration), k: usize) {
    let opts = bench_opts();
    let mut stages = [Duration::MAX; 3];
    let mut worst_total = Duration::ZERO;
    let mut rel = 0f32;
    for _ in 0..REPS {
        let (input, upload_t) = upload();
        let (solve, solve_t) =
            timed(|| nmf_hals_mlx_device(&input, k, &opts, 0).expect("MLX solve"));
        let (res, read_t) = timed(|| solve.read().expect("MLX readback"));
        assert_real_work("mlx", res.final_loss, input.sq_frob());
        rel = res.final_loss / input.sq_frob();
        for (best, d) in stages.iter_mut().zip([upload_t, solve_t, read_t]) {
            *best = (*best).min(d);
        }
        worst_total = worst_total.max(upload_t + solve_t + read_t);
    }
    let [upload, solve, read] = stages;
    println!(
        "  mlx    upload {upload:>9.2?}  solve {solve:>9.2?}  readback {read:>9.2?}  \
         (worst total {worst_total:>9.2?})  rel loss {rel:.6}"
    );
    println!(
        "         solve {:.2} ms/iter",
        solve.as_secs_f64() * 1e3 / FIXED_ITERS as f64
    );
}

/// cubecl solve on wgpu, staged into upload and solve. The solve includes the
/// scratch allocation and the readback, which its public API does not split.
///
/// ### Params
///
/// * `upload` - Builds the device input on the given client
/// * `shape` - The problem, for the scratch size
fn bench_cubecl<In>(
    mut upload: impl FnMut(&cubecl::prelude::Client) -> (In, Duration),
    shape: NmfShape,
) where
    In: bixverse_gpu::methods_gpu::nmf_gpu::GpuNmfData,
{
    use bixverse_gpu::methods_gpu::nmf_gpu::{NmfGpuScratch, nmf_hals_gpu};
    use cubecl::wgpu::WgpuDevice;

    let NmfShape { m, n, k, .. } = shape;
    let opts = bench_opts();
    let client = cubecl::Device::from(WgpuDevice::default()).client();
    let mut stages = [Duration::MAX; 2];
    let mut worst_total = Duration::ZERO;
    let mut rel = 0f32;
    for _ in 0..REPS {
        let (input, upload_t) = upload(&client);
        let (res, solve_t) = timed(|| {
            let scratch = NmfGpuScratch::new(m, n, k, opts.eps, &client).expect("scratch");
            nmf_hals_gpu(&input, k, &opts, &scratch, &client, 0).expect("cubecl solve")
        });
        let sq_frob = input.sq_frob();
        assert_real_work("cubecl", res.final_loss, sq_frob);
        rel = res.final_loss / sq_frob;
        for (best, d) in stages.iter_mut().zip([upload_t, solve_t]) {
            *best = (*best).min(d);
        }
        worst_total = worst_total.max(upload_t + solve_t);
    }
    let [upload, solve] = stages;
    println!(
        "  cubecl upload {upload:>9.2?}  solve+readback {solve:>9.2?}  \
         (worst total {worst_total:>9.2?})  rel loss {rel:.6}"
    );
}

/// Block until every queued cubecl kernel has retired.
///
/// ### Params
///
/// * `client` - CubeCL compute client
fn sync(client: &cubecl::prelude::Client) {
    cubecl::future::block_on(client.sync()).expect("device sync failed");
}

//////////
// Runs //
//////////

/// All arms on one dense shape.
///
/// ### Params
///
/// * `shape` - The problem
fn run_dense(shape: NmfShape) {
    let NmfShape { label, m, n, k, .. } = shape;
    println!("-- {label} dense: {m} x {n}, k = {k}, {FIXED_ITERS} iters --");
    let v = build_dense(m, n, k);

    bench_cpu(&DenseInput::new(v.as_ref()).expect("CPU input"), k);

    bench_mlx(
        || timed(|| MlxDenseNmfInput::new(v.as_ref()).expect("MLX input")),
        k,
    );
    bench_cubecl(
        |client| {
            use bixverse_gpu::methods_gpu::nmf_gpu::GpuDenseNmfInput;
            timed(|| {
                let input = GpuDenseNmfInput::new(v.as_ref(), client).expect("cubecl input");
                sync(client);
                input
            })
        },
        shape,
    );
}

/// All arms on one sparse shape. Host-side validation and the transpose are
/// done outside the upload timing, as in `gpu_nmf_bench`.
///
/// ### Params
///
/// * `shape` - The problem
fn run_sparse(shape: NmfShape) {
    let NmfShape {
        label,
        m,
        n,
        k,
        density,
    } = shape;
    let density = density.expect("sparse shape without a density");
    let (csr, nnz) = build_sparse(m, n, density);
    println!(
        "-- {label} sparse: {m} x {n}, k = {k}, nnz = {nnz} ({:.1}% dense), {FIXED_ITERS} iters --",
        100.0 * nnz as f64 / (m as f64 * n as f64)
    );
    let host = || SparseInput::<f32, f32>::from_primary(&csr).expect("host sparse input");

    bench_cpu(&host(), k);

    bench_mlx(
        || {
            let h = host();
            timed(|| MlxSparseNmfInput::new(h).expect("MLX input"))
        },
        k,
    );
    bench_cubecl(
        |client| {
            use bixverse_gpu::methods_gpu::nmf_gpu::GpuSparseNmfInput;
            let h = host();
            timed(|| {
                let input = GpuSparseNmfInput::new(h, client).expect("cubecl input");
                sync(client);
                input
            })
        },
        shape,
    );
}

//////////
// Main //
//////////

fn main() {
    for shape in SPARSE_SHAPES {
        run_sparse(shape);
    }
    if std::env::var("BIXVERSE_BENCH_BIG").is_ok() {
        for shape in BIG_SHAPES {
            run_sparse(shape);
        }
    } else {
        println!("Set BIXVERSE_BENCH_BIG=1 to also run the 200k-cell sparse shape.");
    }
    if std::env::var("BIXVERSE_BENCH_DENSE").is_ok() {
        for shape in DENSE_SHAPES {
            run_dense(shape);
        }
    } else {
        println!("Set BIXVERSE_BENCH_DENSE=1 to also run the dense bulk and metacell shapes.");
    }
}
