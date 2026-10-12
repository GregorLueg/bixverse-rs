//! Dense HALS NMF: CPU against MLX, plus the cubecl path when `gpu` is on.
//!
//! Same shapes and fixed iteration count as `gpu_nmf_bench`, so the numbers
//! line up. The MLX run is split into its three separable stages: upload of
//! `V` (validation plus the host-to-device copy), the solve (blocks on its last
//! objective, so the device is idle when it returns) and the readback of the
//! factors.
//!
//! Run with:
//! ```text
//! cargo bench --features mlx --bench mlx_nmf_bench
//! cargo bench --features mlx,gpu --bench mlx_nmf_bench
//! ```
//!
//! Needs `libmlxc` (see `build.rs`); without it the `mlx` module does not
//! exist and this does not compile.

use std::time::{Duration, Instant};

use faer::Mat;

use bixverse_rs::methods::nmf_hals::dense::DenseInput;
use bixverse_rs::methods::nmf_hals::{HalsOpts, NmfInit, NmfInput, nmf_hals};
use bixverse_rs::mlx::methods::nmf::{MlxDenseNmfInput, nmf_hals_mlx_device};

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
}

/// The dense shapes of `gpu_nmf_bench` and the `skinny_gemm` doc table.
const SHAPES: [NmfShape; 2] = [
    NmfShape {
        label: "bulk",
        m: 500,
        n: 20_000,
        k: 10,
    },
    NmfShape {
        label: "metacell",
        m: 5_000,
        n: 3_000,
        k: 30,
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
/// * `v` - The matrix
/// * `k` - Components
///
/// ### Returns
///
/// `(best, worst, relative loss)`.
fn bench_cpu(v: &Mat<f32>, k: usize) -> (Duration, Duration, f32) {
    let opts = bench_opts();
    let input = DenseInput::new(v.as_ref()).expect("CPU input");
    let sq_frob = NmfInput::sq_frob(&input);
    let mut loss = 0f32;
    let (best, worst) = best_of(|| {
        let (res, d) = timed(|| nmf_hals(&input, k, &opts, 0).expect("CPU solve"));
        loss = res.final_loss;
        d
    });
    assert_real_work("cpu", loss, sq_frob);
    (best, worst, loss / sq_frob)
}

/// MLX solve, staged.
///
/// ### Params
///
/// * `v` - The matrix
/// * `k` - Components
fn bench_mlx(v: &Mat<f32>, k: usize) {
    let opts = bench_opts();
    let mut stages = [Duration::MAX; 3];
    let mut worst_total = Duration::ZERO;
    let mut rel = 0f32;
    for _ in 0..REPS {
        let (input, upload) = timed(|| MlxDenseNmfInput::new(v.as_ref()).expect("MLX input"));
        let (solve, solve_t) =
            timed(|| nmf_hals_mlx_device(&input, k, &opts, 0).expect("MLX solve"));
        let (res, read_t) = timed(|| solve.read().expect("MLX readback"));
        assert_real_work("mlx", res.final_loss, input.sq_frob());
        rel = res.final_loss / input.sq_frob();
        for (best, d) in stages.iter_mut().zip([upload, solve_t, read_t]) {
            *best = (*best).min(d);
        }
        worst_total = worst_total.max(upload + solve_t + read_t);
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

/// cubecl solve on wgpu, end to end including upload and scratch.
///
/// ### Params
///
/// * `v` - The matrix
/// * `k` - Components
#[cfg(feature = "gpu")]
fn bench_cubecl(v: &Mat<f32>, k: usize) {
    use bixverse_rs::gpu::methods_gpu::nmf_gpu::{
        GpuDenseNmfInput, GpuNmfData, NmfGpuScratch, nmf_hals_gpu,
    };
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    let opts = bench_opts();
    let client = <WgpuRuntime as cubecl::Runtime>::client(&WgpuDevice::DefaultDevice);
    let (m, n) = (v.nrows(), v.ncols());
    let mut rel = 0f32;
    let (best, worst) = best_of(|| {
        // `nmf_hals_gpu` reads the factors back, so it returns with the queue
        // drained.
        let ((loss, sq_frob), d) = timed(|| {
            let input = GpuDenseNmfInput::<WgpuRuntime>::new(v.as_ref(), &client).expect("input");
            let scratch =
                NmfGpuScratch::<WgpuRuntime>::new(m, n, k, opts.eps, &client).expect("scratch");
            let res = nmf_hals_gpu(&input, k, &opts, &scratch, &client, 0).expect("solve");
            (res.final_loss, GpuNmfData::sq_frob(&input))
        });
        assert_real_work("cubecl", loss, sq_frob);
        rel = loss / sq_frob;
        d
    });
    println!("  cubecl {best:>9.2?} (worst {worst:>9.2?})  end to end  rel loss {rel:.6}");
}

//////////
// Main //
//////////

fn main() {
    for shape in SHAPES {
        let NmfShape { label, m, n, k } = shape;
        println!("-- {label} dense: {m} x {n}, k = {k}, {FIXED_ITERS} iters --");
        let v = build_dense(m, n, k);

        let (best, worst, rel) = bench_cpu(&v, k);
        println!("  cpu    {best:>9.2?} (worst {worst:>9.2?})  rel loss {rel:.6}");

        bench_mlx(&v, k);

        #[cfg(feature = "gpu")]
        bench_cubecl(&v, k);
    }
}
