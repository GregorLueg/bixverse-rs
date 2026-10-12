//! End-to-end benchmark for the MLX pairwise correlation path (Pearson).
//!
//! Same shapes and synthetic data as `gpu_corr_bench.rs`. Every shape runs the
//! MLX path end to end (best of N), once more with its stage split (prep,
//! upload, compute, readback), the faer CPU path, and with `gpu` on, the cubecl
//! path. Each result is guarded on its Pearson invariants and the MLX output is
//! compared against the CPU one.
//!
//! Run with:
//! ```text
//! cargo bench --features mlx --bench mlx_corr_bench
//! cargo bench --features mlx,gpu --bench mlx_corr_bench
//! ```
//!
//! `BIXVERSE_BENCH_BIG=1` adds the 20000-column and 200k-row shapes.

use std::time::{Duration, Instant};

use faer::Mat;

use bixverse_rs::core::base::cors_similarity::column_pairwise_cor;
use bixverse_rs::mlx::linalg::corr::{
    MlxCorCov, column_pairwise_cor_mlx, column_pairwise_cor_mlx_staged,
};

////////////
// Shapes //
////////////

/// End-to-end repetitions per shape and arm; the best is reported.
const END_TO_END_REPS: usize = 3;

/// Largest acceptable absolute difference between the MLX and CPU Pearson
/// matrices; matches `REL_TOL` in `tests/gpu_corr.rs` with entries in [-1, 1].
const AGREEMENT_TOL: f32 = 1e-3;

/// One synthetic problem: `n` rows (samples) by `d` columns (features).
#[derive(Clone, Copy, Debug)]
struct CorShape {
    /// Label used in the output
    label: &'static str,
    /// Rows, i.e. the reduction length
    n: usize,
    /// Columns, i.e. the output edge
    d: usize,
}

/// Default shapes, identical to `gpu_corr_bench.rs`.
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

/// Shapes behind `BIXVERSE_BENCH_BIG`, identical to `gpu_corr_bench.rs`.
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

/// Deterministic hash to `[0, 1)`, as in `gpu_corr_bench.rs`.
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

/// Dense `[n, d]` matrix: one shared latent factor plus per-entry noise, as
/// in `gpu_corr_bench.rs`.
///
/// ### Params
///
/// * `shape` - Problem dimensions
///
/// ### Returns
///
/// Column-major `Mat<f32>` of shape `[n, d]`.
fn build_matrix(shape: CorShape) -> Mat<f32> {
    let factor: Vec<f32> = (0..shape.n).map(|i| hash01(i, 0xFEED) - 0.5).collect();
    let loading: Vec<f32> = (0..shape.d).map(|j| hash01(j, 0xBEEF) - 0.5).collect();
    Mat::from_fn(shape.n, shape.d, |i, j| {
        factor[i] * loading[j] + (hash01(i, j.wrapping_add(0xA5A5)) - 0.5)
    })
}

/// Reject a run that did no work: Pearson has a unit diagonal and entries in
/// `[-1, 1]`. The off-diagonal walk uses a stride rather than all `d^2`.
///
/// ### Params
///
/// * `g` - Correlation matrix `[d, d]`
/// * `label` - Shape and arm label, for the panic message
fn guard_pearson(g: &Mat<f32>, label: &str) {
    let d = g.nrows();
    let trace: f64 = (0..d).map(|i| g[(i, i)] as f64).sum();
    let mean_diag = trace / d as f64;
    assert!(
        (mean_diag - 1.0).abs() < 1e-3,
        "{label}: mean Pearson diagonal is {mean_diag:.6}, expected 1.0"
    );
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

/// Max absolute difference between two equal-shape matrices.
///
/// ### Params
///
/// * `a` - First matrix
/// * `b` - Second matrix
///
/// ### Returns
///
/// `max |a - b|` over all entries.
fn max_abs_diff(a: &Mat<f32>, b: &Mat<f32>) -> f32 {
    let mut worst = 0.0f32;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            worst = worst.max((a[(i, j)] - b[(i, j)]).abs());
        }
    }
    worst
}

/// Time `f` `END_TO_END_REPS` times, guard each result, keep the last one.
///
/// ### Params
///
/// * `label` - Shape and arm label, for the guard
/// * `f` - One end-to-end run
///
/// ### Returns
///
/// Best and worst wall time, plus the last result.
fn best_of(label: &str, mut f: impl FnMut() -> Mat<f32>) -> (Duration, Duration, Mat<f32>) {
    let mut times = Vec::with_capacity(END_TO_END_REPS);
    let mut last = Mat::new();
    for _ in 0..END_TO_END_REPS {
        let t = Instant::now();
        let g = f();
        times.push(t.elapsed());
        guard_pearson(&g, label);
        last = g;
    }
    times.sort();
    (times[0], times[END_TO_END_REPS - 1], last)
}

////////////
// Runner //
////////////

/// Run every arm on one shape and print the results.
///
/// ### Params
///
/// * `shape` - Problem dimensions
fn bench_shape(shape: CorShape) {
    let CorShape { label, n, d } = shape;
    let mat = build_matrix(shape);
    let in_mb = (n * d * 4) as f64 / 1e6;
    let out_mb = (d * d * 4) as f64 / 1e6;
    println!("--- {label}: n={n}, d={d}  (in {in_mb:.0} MB, out {out_mb:.0} MB) ---");

    let (best, worst, mlx) = best_of(&format!("{label} mlx"), || {
        column_pairwise_cor_mlx(mat.as_ref(), MlxCorCov::Pearson, false)
            .expect("column_pairwise_cor_mlx failed")
    });
    println!("  MLX end to end     {best:>12.2?}  (best of {END_TO_END_REPS}, worst {worst:.2?})");

    let (g, st) = column_pairwise_cor_mlx_staged(mat.as_ref(), MlxCorCov::Pearson)
        .expect("column_pairwise_cor_mlx_staged failed");
    guard_pearson(&g, label);
    drop(g);
    let tot = st.total().as_secs_f64();
    let pct = |x: Duration| 100.0 * x.as_secs_f64() / tot;
    println!(
        "    prep             {:>12.2?}  {:>5.1}%",
        st.prep,
        pct(st.prep)
    );
    println!(
        "    upload           {:>12.2?}  {:>5.1}%",
        st.upload,
        pct(st.upload)
    );
    println!(
        "    compute          {:>12.2?}  {:>5.1}%",
        st.compute,
        pct(st.compute)
    );
    println!(
        "    readback         {:>12.2?}  {:>5.1}%",
        st.readback,
        pct(st.readback)
    );

    let (best, worst, cpu) = best_of(&format!("{label} cpu"), || {
        column_pairwise_cor(&mat.as_ref(), false)
    });
    println!("  CPU (faer)         {best:>12.2?}  (best of {END_TO_END_REPS}, worst {worst:.2?})");

    let diff = max_abs_diff(&mlx, &cpu);
    assert!(
        diff <= AGREEMENT_TOL,
        "{label}: MLX and CPU disagree by {diff:.3e}"
    );
    println!("    (MLX vs CPU: max abs diff {diff:.2e})");
    drop(cpu);

    #[cfg(feature = "gpu")]
    {
        use bixverse_rs::gpu::linalg::corr::{GpuCorCov, column_pairwise_cor_gpu};
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

        let (best, worst, _) = best_of(&format!("{label} cubecl"), || {
            column_pairwise_cor_gpu::<f32, WgpuRuntime>(
                mat.as_ref(),
                GpuCorCov::Pearson,
                WgpuDevice::DefaultDevice,
                false,
            )
            .expect("column_pairwise_cor_gpu failed")
        });
        println!(
            "  cubecl (wgpu)      {best:>12.2?}  (best of {END_TO_END_REPS}, worst {worst:.2?})"
        );
    }
    println!();
}

fn main() {
    println!("====== MLX pairwise correlation bench (Pearson) ======\n");
    for shape in DEFAULT_SHAPES {
        bench_shape(shape);
    }
    if std::env::var("BIXVERSE_BENCH_BIG").is_ok() {
        for shape in BIG_SHAPES {
            bench_shape(shape);
        }
    } else {
        println!("Set BIXVERSE_BENCH_BIG=1 to also run the 20000-column and 200k-row shapes.");
    }
}
