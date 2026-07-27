//! End-to-end benchmark for the GPU randomised sparse SVD.
//!
//! `randomised_sparse_svd_gpu` has one production caller,
//! `pca_on_sc_sparse_gpu`, which runs it on single-cell counts: cells by
//! genes, CSC, roughly 10% density, `n_components = 30` and an oversampling
//! of 100, so the internal working width is `s = 130`. Nothing in the crate
//! exercised that regime before this file; the three unit tests are all 60x20
//! or smaller.
//!
//! The default shape is 200k x 2000, which keeps a run under a minute and
//! still moves ~4e7 non-zeros. Set `BIXVERSE_BENCH_BIG=1` to add the real
//! 1M x 2000 shape (~2e8 nnz, several GB of VRAM across both layouts).
//!
//! Run with: cargo bench --bench gpu_sparse_svd_bench --features gpu

#![allow(missing_docs)]

use std::time::Instant;

use cubecl::prelude::Runtime;

use bixverse_rs::gpu::linalg::sparse_rand_svd_gpu::{RandSvdGpuParams, randomised_sparse_svd_gpu};
use bixverse_rs::prelude::*;

////////////
// Consts //
////////////

/// Non-zeros per column as a fraction. Matches the density the module doc for
/// `sparse_rand_svd_gpu` designs against.
const DENSITY_DIVISOR: usize = 10;

/// Leading singular triples requested. The single-cell PCA path asks for
/// 30-50; the module doc's worked example uses 30.
const N_COMPONENTS: usize = 30;

/// Oversampling. `pca_on_sc_sparse_gpu` passes `min(100, ...)`, so 100 in
/// anything but a degenerate input. With `N_COMPONENTS` this puts the
/// internal width at 130, which is what every buffer and every byte of
/// traffic in the pipeline scales with.
const OVERSAMPLING: usize = 100;

/// Power iterations. Hard-coded to 2 at the production call site.
const N_POWER_ITERS: usize = 2;

/// Standard-deviation floor. `randomised_sparse_svd_gpu` requires strictly
/// positive column scales and expects the caller to have floored them.
const SIGMA_FLOOR: f32 = 1e-6;

////////////
// Shapes //
////////////

/// One synthetic problem.
#[derive(Clone, Copy, Debug)]
struct SvdShape {
    /// Human-readable label used in the output.
    label: &'static str,
    /// Rows of A, i.e. cells.
    n: usize,
    /// Columns of A, i.e. genes.
    m: usize,
}

const DEFAULT_SHAPE: SvdShape = SvdShape {
    label: "sc-medium",
    n: 200_000,
    m: 2_000,
};

const BIG_SHAPE: SvdShape = SvdShape {
    label: "sc-production",
    n: 1_000_000,
    m: 2_000,
};

/////////////
// Helpers //
/////////////

/// Number of latent factors in the synthetic spectrum. Gives the singular
/// values something to decay along instead of a flat noise floor, while the
/// per-entry noise keeps the matrix full rank. Generation costs `N_LATENT`
/// fused multiply-adds per non-zero, so this stays small: at the production
/// shape 16 already means 3.2e9 of them.
const N_LATENT: usize = 16;

/// Deterministic hash to `[0, 1)`.
///
/// The values must be effectively random per entry, not a low-order function
/// of `(i, j)`. A structured value model makes A rank-deficient, and
/// CholeskyQR2 then fails outright with a non-positive pivot rather than
/// degrading, which is a bench artefact rather than anything about the code
/// under test.
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

/// Build a deterministic CSC with exactly `n / DENSITY_DIVISOR` non-zeros per
/// column.
///
/// Rows are walked with a fixed stride from a per-column offset, so the row
/// indices inside every column segment come out sorted ascending, which is
/// what the SpMM kernels assume. Building by stride rather than by testing
/// every cell keeps this O(nnz) instead of O(n * m); at the production shape
/// the difference is 2e8 against 2e9.
///
/// Values are a decaying low-rank signal plus full-rank per-entry noise, so
/// the spectrum both decays and stays well conditioned at `s = 130`.
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

    // Gene loadings for the latent factors, drawn once.
    let loadings: Vec<f32> = (0..shape.m * N_LATENT)
        .map(|t| hash01(t, 0xFEED) - 0.5)
        .collect();

    // Cell scores, precomputed per row so the inner loop is two loads and an
    // FMA rather than a hash. Amplitude decays with the factor index, giving
    // a spectrum that falls off rather than a plateau.
    let scores: Vec<f32> = (0..shape.n * N_LATENT)
        .map(|t| {
            let f = t % N_LATENT;
            (hash01(t / N_LATENT, f) - 0.5) / (1.0 + f as f32)
        })
        .collect();

    for j in 0..shape.m {
        let load = &loadings[j * N_LATENT..(j + 1) * N_LATENT];
        for t in 0..per_col {
            // One non-zero per block of `DENSITY_DIVISOR` rows, jittered
            // inside its block. Blocks do not overlap, so the segment stays
            // ascending without a sort, and the support differs per column
            // rather than falling into `DENSITY_DIVISOR` shared patterns.
            // Shared supports put block structure into the centred matrix and
            // blow up its condition number, which CholeskyQR2 cannot survive
            // in fp32.
            let jitter = (hash01(j, t.wrapping_add(0xBEEF)) * DENSITY_DIVISOR as f32) as usize;
            let i = t * DENSITY_DIVISOR + jitter.min(DENSITY_DIVISOR - 1);
            indices.push(i as u32);

            let score = &scores[i * N_LATENT..(i + 1) * N_LATENT];
            let mut v = 0.0f32;
            for f in 0..N_LATENT {
                v += score[f] * load[f];
            }
            values.push(v + hash01(i, j.wrapping_add(0xA5A5)) + 0.1);
        }
        indptr.push(values.len() as u32);
    }

    (values, indices, indptr)
}

/// Column means and standard deviations over the full dense column, zeros
/// included, matching what `pca_on_sc_sparse_gpu` computes upstream.
///
/// ### Params
///
/// * `values` - CSC values
/// * `indptr` - CSC column pointers, length `m + 1`
/// * `shape` - Problem dimensions
///
/// ### Returns
///
/// `(means, standard deviations)`, both length `m`, deviations floored above
/// zero.
fn column_stats(values: &[f32], indptr: &[u32], shape: SvdShape) -> (Vec<f32>, Vec<f32>) {
    let inv_n = 1.0 / shape.n as f32;
    let mut means = Vec::with_capacity(shape.m);
    let mut stds = Vec::with_capacity(shape.m);

    for j in 0..shape.m {
        let seg = &values[indptr[j] as usize..indptr[j + 1] as usize];
        // Accumulate in f64: at n = 1e6 the running sum of squares is where
        // f32 cancellation would bite.
        let sum: f64 = seg.iter().map(|&v| v as f64).sum();
        let sum_sq: f64 = seg.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let mean = sum * inv_n as f64;
        let var = (sum_sq * inv_n as f64 - mean * mean).max(0.0);
        means.push(mean as f32);
        stds.push((var.sqrt() as f32).max(SIGMA_FLOOR));
    }

    (means, stds)
}

/// Reject a run that produced no work.
///
/// The pipeline dispatches with `launch_unchecked` throughout, which does
/// nothing and reports no error when a device limit is busted, and the `[n, s]`
/// workspaces come from `GpuTensor::empty`. A dead run therefore returns
/// whatever the host SVD made of uninitialised VRAM, which is overwhelmingly
/// likely to be zero, non-finite, or unsorted.
///
/// ### Params
///
/// * `s` - Singular values returned by the driver
fn guard_singular_values(s: &[f32]) {
    assert_eq!(s.len(), N_COMPONENTS, "wrong number of singular values");
    assert!(
        s.iter().all(|v| v.is_finite()),
        "non-finite singular value: the GPU almost certainly did no work"
    );
    assert!(
        s[0] > 1e-3,
        "leading singular value {} is degenerate: the GPU almost certainly did no work",
        s[0]
    );
    for i in 1..s.len() {
        assert!(
            s[i - 1] >= s[i],
            "singular values not weakly decreasing at {}",
            i
        );
    }
}

////////////
// Runner //
////////////

/// Run one shape and print the timing split between host build, upload plus
/// factorisation, and the guard.
fn run_shape<R: Runtime>(shape: SvdShape, device: &R::Device)
where
    R::Device: Clone,
{
    let s_width = N_COMPONENTS + OVERSAMPLING;
    let build_start = Instant::now();
    let (values, indices, indptr) = build_csc(shape);
    let nnz = values.len();
    let (means, stds) = column_stats(&values, &indptr, shape);
    let build_time = build_start.elapsed();

    println!(
        "--- {}: n={}, m={}, nnz={} ({:.1}%), s={} ---",
        shape.label,
        shape.n,
        shape.m,
        nnz,
        100.0 / DENSITY_DIVISOR as f64,
        s_width
    );
    println!("  host CSC build: {:.2?}", build_time);

    // `CompressedSparseData2` borrows, and `randomised_sparse_svd_gpu`
    // consumes it, so rebuild the view per repetition.
    for rep in 0..2 {
        let csc = CompressedSparseData2::<f32, f32>::new_csc(
            &values,
            &indices,
            &indptr,
            Some(&values),
            (shape.n, shape.m),
        );

        let start = Instant::now();
        let svd = randomised_sparse_svd_gpu::<R, f32, f32>(
            csc,
            &means,
            &stds,
            None,
            N_COMPONENTS,
            Some(RandSvdGpuParams::new(N_POWER_ITERS, OVERSAMPLING)),
            42,
            device.clone(),
            0,
        )
        .expect("randomised sparse SVD failed");
        let elapsed = start.elapsed();

        guard_singular_values(&svd.s);
        println!(
            "  rep {}: {:>10.2?}  (s[0] = {:.4}, s[{}] = {:.4})",
            rep,
            elapsed,
            svd.s[0],
            N_COMPONENTS - 1,
            svd.s[N_COMPONENTS - 1]
        );
    }
    println!();
}

fn main() {
    let device: <cubecl::wgpu::WgpuRuntime as Runtime>::Device = Default::default();

    println!(
        "====== GPU randomised sparse SVD bench ({} components, {} oversampling, {} power iters) ======\n",
        N_COMPONENTS, OVERSAMPLING, N_POWER_ITERS
    );

    run_shape::<cubecl::wgpu::WgpuRuntime>(DEFAULT_SHAPE, &device);

    if std::env::var("BIXVERSE_BENCH_BIG").is_ok() {
        run_shape::<cubecl::wgpu::WgpuRuntime>(BIG_SHAPE, &device);
    } else {
        println!("Set BIXVERSE_BENCH_BIG=1 to also run the 1M x 2000 production shape.");
    }
}
