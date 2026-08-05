//! End-to-end benchmark for the GPU pairwise correlation / covariance path.
//!
//! `column_pairwise_cor_gpu` reduces to one product, `G = A A^T` over the
//! centred and scaled matrix, which is uploaded feature-major. That product goes
//! through `gram_aat`. It used to go through cubek with `Strategy::DoubleUnit`,
//! pinned there because `Strategy::Auto` blows up on Apple devices, and
//! `DoubleUnit` gives one thread per output element with a serial reduction:
//! the same wrong-algorithm-for-the-shape problem that cost the randomised SVD
//! 25.7x before it got a dedicated Gram kernel.
//!
//! Both still run here. The cubek arm is timed into `Stages::product_cubek` and
//! left out of `Stages::total`, so every shape scores the replacement against
//! the thing it replaced in one pass rather than against a number from an old
//! run on a different machine.
//!
//! Nothing in the crate benchmarked this path before. The two things this file
//! has to establish are the baseline itself and, per the k-means experience,
//! **where** the time actually goes: a single end-to-end number hides the host
//! side completely, and the ratio inverts as soon as the kernels improve. So
//! every shape is run twice, once end to end with the queue left to pipeline
//! and once stage by stage with a device sync between stages.
//!
//! The two do not add up, and that is expected rather than a bug. The staged
//! run measures isolated stage cost; the end-to-end run measures a pipelined
//! queue.
//!
//! Run with: cargo bench --bench gpu_corr_bench --features gpu

#![allow(missing_docs)]

use std::time::{Duration, Instant};

use cubecl::future;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use faer::Mat;

use cubecl_utils_rs::prelude::*;

use bixverse_rs::gpu::linalg::cholesky_gpu::dense_gemm;
use bixverse_rs::gpu::linalg::corr::{GpuCorCov, column_pairwise_cor_gpu, scale_matrix_col_gpu};
use bixverse_rs::gpu::linalg::gram::gram_aat;
use cubek::matmul::launch::Strategy;

////////////
// Shapes //
////////////

/// End-to-end repetitions per shape. The reported figure is the best of these;
/// see the note in `run_shape`.
const END_TO_END_REPS: usize = 3;

/// One synthetic problem. `n` rows (samples, cells) by `d` columns (features,
/// genes); the output is always `[d, d]`.
#[derive(Clone, Copy, Debug)]
struct CorShape {
    /// Human-readable label used in the output.
    label: &'static str,
    /// Rows, i.e. the reduction length.
    n: usize,
    /// Columns, i.e. the output edge.
    d: usize,
}

/// Default shapes, one per regime. Sized so the whole file runs in a couple of
/// minutes and the largest device buffer stays a few hundred MB.
///
/// The regimes matter because they stress opposite things. `fat` has a huge
/// output and a short reduction, `tall` the reverse, `narrow` has too few
/// output tiles to fill the device at all, and `square` is what the existing
/// diagnostic sweep in `tests/gpu_corr.rs` already covers.
const DEFAULT_SHAPES: [CorShape; 4] = [
    CorShape {
        label: "fat",
        n: 500,
        d: 8_000,
    },
    CorShape {
        label: "tall",
        n: 50_000,
        d: 2_000,
    },
    CorShape {
        label: "square",
        n: 2_000,
        d: 2_000,
    },
    CorShape {
        label: "narrow",
        n: 1_000_000,
        d: 100,
    },
];

/// Shapes behind `BIXVERSE_BENCH_BIG`. `fat-big` alone needs 1.6 GB on the
/// device for the output and another 3.2 GB on the host across `result_flat`
/// and the returned `Mat`, so it is not something to run by accident.
const BIG_SHAPES: [CorShape; 2] = [
    CorShape {
        label: "fat-big",
        n: 500,
        d: 20_000,
    },
    CorShape {
        label: "tall-big",
        n: 200_000,
        d: 2_000,
    },
];

/////////////
// Helpers //
/////////////

/// Deterministic hash to `[0, 1)`.
///
/// Values must be effectively random per entry. A structured value model makes
/// columns collinear, which drives correlations to +/-1 and would make the
/// trace guard below pass for the wrong reason.
///
/// ### Params
///
/// * `a` - First mixing input
/// * `b` - Second mixing input
///
/// ### Returns
///
/// A value in `[0, 1)`.
#[inline]
fn hash01(a: usize, b: usize) -> f32 {
    let mut h = (a as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    ((h >> 40) as f32) / 16_777_216.0
}

/// Build a dense `[n, d]` matrix with a mild shared signal on top of per-entry
/// noise.
///
/// The shared component keeps off-diagonal correlations away from zero, so the
/// output is not a near-identity matrix that any broken kernel could reproduce
/// by accident. It stays small enough that no column pair is degenerate.
///
/// ### Params
///
/// * `shape` - Problem dimensions
///
/// ### Returns
///
/// Column-major `Mat<f32>` of shape `[n, d]`.
fn build_matrix(shape: CorShape) -> Mat<f32> {
    // One latent factor per row, shared across all columns with a per-column
    // loading. Cheap, and enough to put structure in the correlation matrix.
    let factor: Vec<f32> = (0..shape.n).map(|i| hash01(i, 0xFEED) - 0.5).collect();
    let loading: Vec<f32> = (0..shape.d).map(|j| hash01(j, 0xBEEF) - 0.5).collect();

    Mat::from_fn(shape.n, shape.d, |i, j| {
        factor[i] * loading[j] + (hash01(i, j.wrapping_add(0xA5A5)) - 0.5)
    })
}

/// Reject a run that produced no work.
///
/// A Pearson correlation matrix has an exact unit diagonal and off-diagonals
/// bounded in `[-1, 1]`, so `trace(G) == d` plus that bound is a free invariant
/// no partially-working kernel satisfies. It is the specific guard this path
/// needs: everything dispatches with `launch_unchecked`, which does nothing,
/// returns zeros and reports no error when a device limit is busted, and an
/// implausibly fast run is that failure far more often than it is a real
/// speedup.
///
/// The off-diagonal check walks a stride rather than all `d^2` entries, which
/// at `d = 20000` would be 4e8 comparisons for no extra confidence.
///
/// ### Params
///
/// * `g` - Correlation matrix `[d, d]`
/// * `label` - Shape label, for the panic message
fn guard_pearson(g: &Mat<f32>, label: &str) {
    let d = g.nrows();
    let trace: f64 = (0..d).map(|i| g[(i, i)] as f64).sum();
    let mean_diag = trace / d as f64;
    assert!(
        (mean_diag - 1.0).abs() < 1e-3,
        "{label}: mean Pearson diagonal is {mean_diag:.6}, expected 1.0. \
         The GPU almost certainly did no work."
    );

    // Coprime-ish stride so the walk covers distinct (i, j) pairs.
    let mut worst = 0.0f32;
    for i in 0..d {
        let v = g[(i, (i * 7 + 3) % d)];
        assert!(v.is_finite(), "{label}: non-finite entry at ({i}, ..)");
        worst = worst.max(v.abs());
    }
    assert!(
        worst <= 1.0 + 1e-4,
        "{label}: off-diagonal correlation {worst} is outside [-1, 1]"
    );
}

/// Block until every queued kernel has retired.
///
/// ### Params
///
/// * `client` - CubeCL compute client
fn sync<R: Runtime>(client: &ComputeClient<R>) {
    future::block_on(client.sync()).expect("device sync failed");
}

/////////////////////
// Staged breakdown //
/////////////////////

/// Per-stage timings for one run, in pipeline order.
#[derive(Default)]
struct Stages {
    /// Host-side flatten of the faer matrix into a feature-major buffer.
    flatten: Duration,
    /// Upload to the device.
    upload: Duration,
    /// `column_stats` plus `apply_centre_scale`.
    scale: Duration,
    /// The Gram product via `gram_aat`.
    product: Duration,
    /// The same product via cubek's `DoubleUnit`, for comparison only.
    product_cubek: Duration,
    /// Readback of the `[d, d]` output.
    readback: Duration,
    /// Host-side assembly of the returned `Mat`.
    assemble: Duration,
}

impl Stages {
    /// Total across every stage, counting only the `gram_aat` product.
    fn total(&self) -> Duration {
        self.flatten + self.upload + self.scale + self.product + self.readback + self.assemble
    }
}

/// Re-run the Pearson pipeline stage by stage with a device sync between each,
/// running both products so the replacement is timed and checked against the
/// thing it replaces in one pass.
///
/// Mirrors `column_pairwise_cor_gpu` exactly rather than calling it, because
/// the entry point is one opaque call and the point here is the split. The
/// syncs make each number an isolated stage cost, which is what attribution
/// needs and is deliberately not what the end-to-end run measures.
///
/// ### Params
///
/// * `mat` - Input matrix `[n, d]`
/// * `shape` - Problem dimensions
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// Per-stage timings, plus the assembled matrix so the caller can guard it.
fn run_staged<R: Runtime>(
    mat: &Mat<f32>,
    shape: CorShape,
    client: &ComputeClient<R>,
) -> (Stages, Mat<f32>) {
    let CorShape { n, d, label } = shape;
    let mut st = Stages::default();

    let t = Instant::now();
    let mut data_flat: Vec<f32> = Vec::with_capacity(n * d);
    for j in 0..d {
        data_flat.extend(mat.col(j).iter().cloned());
    }
    st.flatten = t.elapsed();

    let t = Instant::now();
    let data_gpu = GpuTensor::<R, f32>::from_slice(&data_flat, vec![d, n], client).unwrap();
    sync(client);
    st.upload = t.elapsed();

    let t = Instant::now();
    let scaled = scale_matrix_col_gpu(&data_gpu, n, d, true, client).expect("scale failed");
    sync(client);
    st.scale = t.elapsed();

    // Both output buffers are allocated and written once before either is
    // timed. `client.empty()` returns quickly but the device pages are not
    // backed until something writes them, and at d = 8000 that first touch is
    // 256 MB: leaving it inside the timed region attributes a page-fault cost
    // to whichever kernel happened to run first and moved the same config by
    // 1.6x between runs.
    let result = GpuTensor::<R, f32>::empty(vec![d, d], client).unwrap();
    let baseline = GpuTensor::<R, f32>::empty(vec![d, d], client).unwrap();

    let run_gram = || {
        gram_aat::<R, f32>(client, &scaled, &result, n, d).expect("gram_aat failed");
        sync(client);
    };
    run_gram();
    let t = Instant::now();
    run_gram();
    st.product = t.elapsed();

    let run_cubek = || {
        dense_gemm::<R, f32>(
            scaled.handle(),
            [d, n],
            false,
            scaled.handle(),
            [n, d],
            true,
            baseline.handle(),
            [d, d],
            Some(Strategy::DoubleUnit(Default::default())),
            client,
        )
        .expect("dense_gemm failed");
        sync(client);
    };
    run_cubek();
    let t = Instant::now();
    run_cubek();
    st.product_cubek = t.elapsed();

    let t = Instant::now();
    let flat = result.read(client).expect("readback failed");
    st.readback = t.elapsed();

    // A speedup claim against a kernel that disagrees with the reference is
    // worth nothing, and an all-zeros output from a busted dispatch is the
    // single most likely way to "win" here.
    let want = baseline.read(client).expect("baseline readback failed");
    check_against_baseline(&flat, &want, d, label);

    let t = Instant::now();
    let out = Mat::from_fn(d, d, |i, j| flat[j * d + i]);
    st.assemble = t.elapsed();

    (st, out)
}

/// Compare the two products elementwise over every entry.
///
/// Both accumulate the same `n` products in fp32 but in a different order, so
/// this is a relative tolerance rather than a bitwise check. At `n = 1e6` the
/// running sum has a relative error around `sqrt(n) * eps ~ 6e-5`, which is
/// what the bound allows for.
///
/// ### Params
///
/// * `got` - Output of `gram_aat`, row-major `[d, d]`
/// * `want` - Output of cubek's GEMM, row-major `[d, d]`
/// * `d` - Output edge
/// * `label` - Shape label, for the panic message
fn check_against_baseline(got: &[f32], want: &[f32], d: usize, label: &str) {
    let mut worst = 0.0f32;
    let mut at = (0usize, 0usize);
    for i in 0..d {
        for j in 0..d {
            let (a, b) = (got[i * d + j], want[i * d + j]);
            let rel = (a - b).abs() / b.abs().max(1e-3);
            if rel > worst {
                worst = rel;
                at = (i, j);
            }
        }
    }
    assert!(
        worst < 1e-3,
        "{label}: gram_aat disagrees with cubek by {worst:.3e} at {at:?} \
         (got {}, want {})",
        got[at.0 * d + at.1],
        want[at.0 * d + at.1]
    );
    println!("    (agreement with cubek: worst relative diff {worst:.2e})");
}

////////////
// Runner //
////////////

/// Run one shape end to end and staged, and print both.
///
/// ### Params
///
/// * `shape` - Problem dimensions
/// * `device` - CubeCL device
fn run_shape<R: Runtime>(shape: CorShape, device: &R::Device)
where
    R::Device: Clone,
{
    let CorShape { label, n, d } = shape;
    let build_start = Instant::now();
    let mat = build_matrix(shape);
    let build = build_start.elapsed();

    let in_mb = (n * d * 4) as f64 / 1e6;
    let out_mb = (d * d * 4) as f64 / 1e6;
    println!("--- {label}: n={n}, d={d}  (in {in_mb:.0} MB, out {out_mb:.0} MB) ---");
    println!("  synthetic build:   {build:>12.2?}");

    // End to end, queue left to pipeline. This is the number that matters.
    //
    // Best of N rather than a single shot. Run to run this spreads by ~10% on
    // the shapes dominated by large host allocations, which is wide enough to
    // swallow a real change, and the first rep additionally carries shader
    // compilation and the buffer pool's first touch.
    let mut times = Vec::with_capacity(END_TO_END_REPS);
    for _ in 0..END_TO_END_REPS {
        let t = Instant::now();
        let g = column_pairwise_cor_gpu::<f32, R>(
            mat.as_ref(),
            GpuCorCov::Pearson,
            device.clone(),
            false,
        )
        .expect("column_pairwise_cor_gpu failed");
        times.push(t.elapsed());
        guard_pearson(&g, label);
        drop(g);
    }
    times.sort();
    println!(
        "  END TO END:        {:>12.2?}  (best of {END_TO_END_REPS}, worst {:.2?})",
        times[0],
        times[END_TO_END_REPS - 1]
    );

    let client = R::client(device);
    let (st, g) = run_staged::<R>(&mat, shape, &client);
    guard_pearson(&g, label);
    drop(g);

    let tot = st.total().as_secs_f64();
    let pct = |x: Duration| 100.0 * x.as_secs_f64() / tot;
    println!(
        "  staged (isolated, sums to {:.2?}):",
        Duration::from_secs_f64(tot)
    );
    println!(
        "    host flatten     {:>12.2?}  {:>5.1}%",
        st.flatten,
        pct(st.flatten)
    );
    println!(
        "    upload           {:>12.2?}  {:>5.1}%",
        st.upload,
        pct(st.upload)
    );
    println!(
        "    stats + scale    {:>12.2?}  {:>5.1}%",
        st.scale,
        pct(st.scale)
    );
    println!(
        "    product (gram)   {:>12.2?}  {:>5.1}%",
        st.product,
        pct(st.product)
    );
    println!(
        "      vs cubek       {:>12.2?}          ({:.2}x)",
        st.product_cubek,
        st.product_cubek.as_secs_f64() / st.product.as_secs_f64()
    );
    println!(
        "    readback         {:>12.2?}  {:>5.1}%",
        st.readback,
        pct(st.readback)
    );
    println!(
        "    host assemble    {:>12.2?}  {:>5.1}%",
        st.assemble,
        pct(st.assemble)
    );
    println!();
}

fn main() {
    let device: <WgpuRuntime as Runtime>::Device = Default::default();

    println!("====== GPU pairwise correlation bench (Pearson) ======\n");

    for shape in DEFAULT_SHAPES {
        run_shape::<WgpuRuntime>(shape, &device);
    }

    if std::env::var("BIXVERSE_BENCH_BIG").is_ok() {
        for shape in BIG_SHAPES {
            run_shape::<WgpuRuntime>(shape, &device);
        }
    } else {
        println!("Set BIXVERSE_BENCH_BIG=1 to also run the 20000-column and 200k-row shapes.");
    }
}
