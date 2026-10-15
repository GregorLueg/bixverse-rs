//! End-to-end benchmark for the MLX randomised sparse SVD.
//!
//! Same shapes and synthetic data as `gpu_sparse_svd_bench.rs`: cells by
//! genes, CSC, 10% density, 30 components, oversampling 100 (`s = 130`), two
//! power iterations. Every shape runs the MLX path end to end, once more with
//! its stage split (prep, upload, SpMM forward, SpMM transpose, QR,
//! finalise), the CPU `randomised_sparse_svd`, and with `gpu` on, the cubecl
//! path. Each result is guarded and the MLX singular values are compared
//! against the CPU ones.
//!
//! Run with:
//! ```text
//! cargo bench --features mlx --bench mlx_sparse_svd_bench
//! cargo bench --features mlx,gpu --bench mlx_sparse_svd_bench
//! ```
//!
//! `BIXVERSE_BENCH_BIG=1` adds the 1M x 2000 production shape.

use std::time::{Duration, Instant};

use bixverse_core::core::math::pca_svd::randomised_sparse_svd;
use bixverse_mlx::linalg::rand_svd::{
    RandSvdMlxParams, randomised_sparse_svd_mlx, randomised_sparse_svd_mlx_staged,
};
use bixverse_mlx::prelude::*;

///////////////
// Constants //
///////////////

/// One row in this many is non-zero per column, i.e. 10% density.
const DENSITY_DIVISOR: usize = 10;

/// Leading singular triples requested.
const N_COMPONENTS: usize = 30;

/// Oversampling, as `pca_on_sc_sparse_gpu` passes it.
const OVERSAMPLING: usize = 100;

/// Power iterations, as at the production call site.
const N_POWER_ITERS: usize = 2;

/// Standard-deviation floor; the drivers require strictly positive scales.
const SIGMA_FLOOR: f32 = 1e-6;

/// Seed for the random range finder.
const SEED: u64 = 42;

/// Timed end-to-end repetitions per GPU arm. Each is printed; the first
/// carries kernel compilation.
const REPS: usize = 3;

/// Largest acceptable relative difference between the MLX and CPU singular
/// values. Both are randomised with different sketches, so this is the
/// randomised-SVD tolerance, not a bitwise one.
const AGREEMENT_TOL: f32 = 1e-2;

/// Number of latent factors in the synthetic spectrum.
const N_LATENT: usize = 16;

////////////
// Shapes //
////////////

/// One synthetic problem.
#[derive(Clone, Copy, Debug)]
struct SvdShape {
    /// Label used in the output
    label: &'static str,
    /// Rows of A, i.e. cells
    n: usize,
    /// Columns of A, i.e. genes
    m: usize,
}

/// Default shape, identical to `gpu_sparse_svd_bench.rs`.
const DEFAULT_SHAPE: SvdShape = SvdShape {
    label: "sc-medium",
    n: 200_000,
    m: 2_000,
};

/// Production shape behind `BIXVERSE_BENCH_BIG`, identical to
/// `gpu_sparse_svd_bench.rs`.
const BIG_SHAPE: SvdShape = SvdShape {
    label: "sc-production",
    n: 1_000_000,
    m: 2_000,
};

/////////////
// Helpers //
/////////////

/// Deterministic hash to `[0, 1)`, as in `gpu_sparse_svd_bench.rs`.
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

/// Deterministic CSC with `n / DENSITY_DIVISOR` non-zeros per column, a
/// decaying low-rank signal plus full-rank noise. Identical to the builder in
/// `gpu_sparse_svd_bench.rs`.
///
/// ### Params
///
/// * `shape` - Problem dimensions
///
/// ### Returns
///
/// `(values, row indices, column pointers)` in CSC order.
fn build_csc(shape: SvdShape) -> (Vec<f32>, Vec<u32>, Vec<u32>) {
    let per_col = shape.n / DENSITY_DIVISOR;
    let nnz = per_col * shape.m;
    let mut values = Vec::with_capacity(nnz);
    let mut indices = Vec::with_capacity(nnz);
    let mut indptr = Vec::with_capacity(shape.m + 1);
    indptr.push(0u32);

    let loadings: Vec<f32> = (0..shape.m * N_LATENT)
        .map(|t| hash01(t, 0xFEED) - 0.5)
        .collect();
    let scores: Vec<f32> = (0..shape.n * N_LATENT)
        .map(|t| {
            let f = t % N_LATENT;
            (hash01(t / N_LATENT, f) - 0.5) / (1.0 + f as f32)
        })
        .collect();

    for j in 0..shape.m {
        let load = &loadings[j * N_LATENT..(j + 1) * N_LATENT];
        for t in 0..per_col {
            let jitter = (hash01(j, t.wrapping_add(0xBEEF)) * DENSITY_DIVISOR as f32) as usize;
            let i = t * DENSITY_DIVISOR + jitter.min(DENSITY_DIVISOR - 1);
            indices.push(i as u32);
            let score = &scores[i * N_LATENT..(i + 1) * N_LATENT];
            let v: f32 = score.iter().zip(load).map(|(s, l)| s * l).sum();
            values.push(v + hash01(i, j.wrapping_add(0xA5A5)) + 0.1);
        }
        indptr.push(values.len() as u32);
    }
    (values, indices, indptr)
}

/// Column means and standard deviations over the full dense column, zeros
/// included, accumulated in f64.
///
/// ### Params
///
/// * `values` - CSC values
/// * `indptr` - CSC column pointers, length `m + 1`
/// * `shape` - Problem dimensions
///
/// ### Returns
///
/// `(means, standard deviations)`, both length `m`, deviations floored.
fn column_stats(values: &[f32], indptr: &[u32], shape: SvdShape) -> (Vec<f32>, Vec<f32>) {
    let inv_n = 1.0 / shape.n as f64;
    (0..shape.m)
        .map(|j| {
            let seg = &values[indptr[j] as usize..indptr[j + 1] as usize];
            let sum: f64 = seg.iter().map(|&v| v as f64).sum();
            let sum_sq: f64 = seg.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let mean = sum * inv_n;
            let var = (sum_sq * inv_n - mean * mean).max(0.0);
            (mean as f32, (var.sqrt() as f32).max(SIGMA_FLOOR))
        })
        .unzip()
}

/// Reject a run that produced no work: the singular values must be finite,
/// non-degenerate and weakly decreasing.
///
/// ### Params
///
/// * `s` - Singular values returned by the driver
/// * `label` - Arm label, for the panic message
fn guard_singular_values(s: &[f32], label: &str) {
    assert_eq!(s.len(), N_COMPONENTS, "{label}: wrong number of values");
    assert!(
        s.iter().all(|v| v.is_finite()),
        "{label}: non-finite singular value"
    );
    assert!(
        s[0] > 1e-3,
        "{label}: leading singular value {} is degenerate",
        s[0]
    );
    for i in 1..s.len() {
        assert!(s[i - 1] >= s[i], "{label}: not decreasing at {i}");
    }
}

/// Worst relative difference between two singular-value vectors.
///
/// ### Params
///
/// * `a` - Values under test
/// * `b` - Reference values
///
/// ### Returns
///
/// `max |a - b| / |b|`.
fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0f32, |w, (x, y)| w.max((x - y).abs() / y.abs()))
}

/// Print one stage row.
///
/// ### Params
///
/// * `name` - Stage name
/// * `d` - Stage time
/// * `total` - Sum of all stages
fn stage_row(name: &str, d: Duration, total: Duration) {
    println!(
        "    {name:<16} {d:>12.2?}  {:>5.1}%",
        100.0 * d.as_secs_f64() / total.as_secs_f64()
    );
}

////////////
// Runner //
////////////

/// Run every arm on one shape and print the results.
///
/// ### Params
///
/// * `shape` - Problem dimensions
fn bench_shape(shape: SvdShape) {
    let t = Instant::now();
    let (values, indices, indptr) = build_csc(shape);
    let (means, stds) = column_stats(&values, &indptr, shape);
    let csc = || {
        CompressedSparseData2::<f32, f32>::new_csc(
            &values,
            &indices,
            &indptr,
            Some(&values),
            (shape.n, shape.m),
        )
    };
    println!(
        "--- {}: n={}, m={}, nnz={}, s={} ---",
        shape.label,
        shape.n,
        shape.m,
        values.len(),
        N_COMPONENTS + OVERSAMPLING
    );
    println!("  host CSC build     {:>12.2?}", t.elapsed());
    let params = Some(RandSvdMlxParams::new(N_POWER_ITERS, OVERSAMPLING));

    let mut mlx_s = Vec::new();
    for rep in 0..REPS {
        let t = Instant::now();
        let svd =
            randomised_sparse_svd_mlx(csc(), &means, &stds, None, N_COMPONENTS, params, SEED, 0)
                .expect("randomised_sparse_svd_mlx failed");
        let elapsed = t.elapsed();
        guard_singular_values(&svd.s, "mlx");
        println!(
            "  MLX rep {rep}          {elapsed:>12.2?}  (s[0] = {:.4})",
            svd.s[0]
        );
        mlx_s = svd.s;
    }

    let (svd, st) =
        randomised_sparse_svd_mlx_staged(csc(), &means, &stds, None, N_COMPONENTS, params, SEED)
            .expect("randomised_sparse_svd_mlx_staged failed");
    guard_singular_values(&svd.s, "mlx staged");
    let total = st.total();
    println!("  MLX staged         {total:>12.2?}");
    stage_row("prep", st.prep, total);
    stage_row("upload", st.upload, total);
    stage_row("spmm forward", st.spmm_forward, total);
    stage_row("spmm transpose", st.spmm_transpose, total);
    stage_row("qr", st.qr, total);
    stage_row("finalise", st.finalise, total);

    let t = Instant::now();
    let cpu = randomised_sparse_svd::<f32, f32>(
        csc(),
        N_COMPONENTS,
        SEED,
        true,
        Some(OVERSAMPLING),
        Some(N_POWER_ITERS),
        Some(&means),
        Some(&stds),
        None,
    )
    .expect("randomised_sparse_svd failed");
    println!("  CPU                {:>12.2?}", t.elapsed());
    // The CPU driver returns every sketch singular value, not just the top
    // `N_COMPONENTS`.
    let cpu_s = &cpu.s[..N_COMPONENTS];
    guard_singular_values(cpu_s, "cpu");
    let diff = max_rel_diff(&mlx_s, cpu_s);
    assert!(diff <= AGREEMENT_TOL, "MLX and CPU disagree by {diff:.3e}");
    println!("    (MLX vs CPU: max rel diff in s {diff:.2e})");
    {
        use bixverse_gpu::linalg::sparse_rand_svd_gpu::{
            RandSvdGpuParams, randomised_sparse_svd_gpu,
        };
        use cubecl::wgpu::WgpuDevice;

        for rep in 0..REPS {
            let t = Instant::now();
            let svd = randomised_sparse_svd_gpu::<f32, f32>(
                csc(),
                &means,
                &stds,
                None,
                N_COMPONENTS,
                Some(RandSvdGpuParams::new(N_POWER_ITERS, OVERSAMPLING)),
                SEED,
                cubecl::Device::from(WgpuDevice::default()),
                0,
            )
            .expect("randomised_sparse_svd_gpu failed");
            let elapsed = t.elapsed();
            guard_singular_values(&svd.s, "cubecl");
            println!(
                "  cubecl rep {rep}       {elapsed:>12.2?}  (s[0] = {:.4})",
                svd.s[0]
            );
        }
    }
    println!();
}

fn main() {
    println!(
        "====== MLX randomised sparse SVD bench ({N_COMPONENTS} components, \
         {OVERSAMPLING} oversampling, {N_POWER_ITERS} power iters) ======\n"
    );
    bench_shape(DEFAULT_SHAPE);
    if std::env::var("BIXVERSE_BENCH_BIG").is_ok() {
        bench_shape(BIG_SHAPE);
    } else {
        println!("Set BIXVERSE_BENCH_BIG=1 to also run the 1M x 2000 production shape.");
    }
}
