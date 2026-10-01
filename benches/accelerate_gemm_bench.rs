//! faer against `ann_search_rs::utils::gemm::gemm` (Accelerate on macOS) at the
//! GEMM shapes bixverse actually runs.
//!
//! * `nmf` - `W^T V` and `V H^T` from the dense HALS loop, top-level `Par::Rayon`.
//! * `cov` - `X^T X`. faer gets the lower-triangle product the covariance code
//!   uses; Accelerate has to do the full square, i.e. twice the flops.
//!   Also cblas `?syrk`, which only fills the lower triangle (macOS).
//! * `tiled` - `Par::Seq` GEMMs inside a rayon loop, to see whether Accelerate's
//!   own threads oversubscribe. Compare runs with `VECLIB_MAXIMUM_THREADS=1`.
//!
//! Every cell reports the max relative difference to faer, so a layout bug
//! cannot pass as a speed-up.
//!
//! Run with:
//! ```
//! cargo bench --bench accelerate_gemm_bench
//! ```
//!
//! `ACCEL_BENCH_LARGE=1` switches to the larger shapes.

use std::hint::black_box;
use std::time::{Duration, Instant};

use bixverse_rs::prelude::*;
use bixverse_rs::utils::faer_parallelism;
use bixverse_rs::utils::gemm::gemm;
use faer::linalg::matmul::matmul;
use faer::linalg::matmul::triangular::{BlockStructure, matmul as triangular_matmul};
use faer::{Accum, Mat, MatRef, Par};
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;

/// Timed repetitions per cell, after one warm-up.
const REPS: usize = 5;

/// Rows per tile in the `tiled` cell.
const TILE_ROWS: usize = 1024;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_ssyrk(
        order: i32,
        uplo: i32,
        trans: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
    fn cblas_dsyrk(
        order: i32,
        uplo: i32,
        trans: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );
}

/// `C = X^T X`, lower triangle, via Accelerate cblas syrk
trait Syrk: BixverseFloat {
    /// ### Params
    ///
    /// * `x` - Column-major `n x p` input
    /// * `c` - Column-major `p x p` output, lower triangle written
    fn syrk_xtx(x: MatRef<Self>, c: &mut Mat<Self>);
}

macro_rules! impl_syrk {
    ($t:ty, $f:ident) => {
        impl Syrk for $t {
            fn syrk_xtx(x: MatRef<$t>, c: &mut Mat<$t>) {
                const COL_MAJOR: i32 = 102;
                const LOWER: i32 = 122;
                const TRANS: i32 = 112;
                let (n, p) = (x.nrows() as i32, x.ncols() as i32);
                let ldc = c.col_stride() as i32;
                #[cfg(target_os = "macos")]
                unsafe {
                    $f(
                        COL_MAJOR,
                        LOWER,
                        TRANS,
                        p,
                        n,
                        1.0,
                        x.as_ptr(),
                        x.col_stride() as i32,
                        0.0,
                        c.as_mut().as_ptr_mut(),
                        ldc,
                    );
                }
            }
        }
    };
}

impl_syrk!(f32, cblas_ssyrk);
impl_syrk!(f64, cblas_dsyrk);

/// Random matrix with entries in [0, 1)
///
/// ### Params
///
/// * `nrows` - Number of rows
/// * `ncols` - Number of columns
/// * `seed` - RNG seed
///
/// ### Returns
///
/// Column-major `nrows x ncols` matrix.
fn random_mat<T: BixverseFloat>(nrows: usize, ncols: usize, seed: u64) -> Mat<T> {
    let mut rng = StdRng::seed_from_u64(seed);
    Mat::from_fn(nrows, ncols, |_, _| {
        T::from_f64(rng.random::<f64>()).unwrap()
    })
}

/// Median wall clock of `REPS` runs after one warm-up
///
/// ### Params
///
/// * `f` - Closure to time
///
/// ### Returns
///
/// Median duration.
fn median_time(mut f: impl FnMut()) -> Duration {
    f();
    let mut times: Vec<Duration> = (0..REPS)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .collect();
    times.sort();
    times[REPS / 2]
}

/// Max relative difference between two matrices, optionally on the lower
/// triangle only
///
/// ### Params
///
/// * `a` - Reference
/// * `b` - Candidate
/// * `lower_only` - Compare only `i >= j`
///
/// ### Returns
///
/// `max |a - b| / max |a|`.
fn max_rel_diff<T: BixverseFloat>(a: MatRef<T>, b: MatRef<T>, lower_only: bool) -> f64 {
    let (mut diff, mut scale) = (0.0f64, 0.0f64);
    for j in 0..a.ncols() {
        let start = if lower_only { j } else { 0 };
        for i in start..a.nrows() {
            let x = a[(i, j)].to_f64().unwrap();
            let y = b[(i, j)].to_f64().unwrap();
            diff = diff.max((x - y).abs());
            scale = scale.max(x.abs());
        }
    }
    diff / scale.max(f64::MIN_POSITIVE)
}

/// Print one result row
///
/// ### Params
///
/// * `label` - Cell description
/// * `faer` - faer median time
/// * `accel` - gemm median time
/// * `rel` - Max relative difference
fn report(label: &str, faer: Duration, accel: Duration, rel: f64) {
    println!(
        "{label:<44} base {:>9.2?}  new {:>9.2?}  speed-up {:>5.2}x  rel diff {rel:.1e}",
        faer,
        accel,
        faer.as_secs_f64() / accel.as_secs_f64()
    );
}

/// NMF HALS products `W^T V` and `V H^T`
///
/// ### Params
///
/// * `n` - Rows of V
/// * `m` - Columns of V
/// * `k` - Rank
/// * `tag` - Float type label
fn bench_nmf<T: BixverseFloat>(n: usize, m: usize, k: usize, tag: &str) {
    let v = random_mat::<T>(n, m, 1);
    let w = random_mat::<T>(n, k, 2);
    let h = random_mat::<T>(k, m, 3);
    let par = faer_parallelism();

    let mut ref_out = Mat::<T>::zeros(k, m);
    let mut out = Mat::<T>::zeros(k, m);
    let tf = median_time(|| {
        matmul(
            ref_out.as_mut(),
            Accum::Replace,
            w.transpose(),
            v.as_ref(),
            T::one(),
            par,
        );
        black_box(&ref_out);
    });
    let ta = median_time(|| {
        gemm(
            out.as_mut(),
            Accum::Replace,
            w.transpose(),
            v.as_ref(),
            T::one(),
            par,
        );
        black_box(&out);
    });
    let rel = max_rel_diff(ref_out.as_ref(), out.as_ref(), false);
    report(&format!("nmf W^T V  {tag} {n}x{m} k={k}"), tf, ta, rel);

    let mut ref_out = Mat::<T>::zeros(n, k);
    let mut out = Mat::<T>::zeros(n, k);
    let tf = median_time(|| {
        matmul(
            ref_out.as_mut(),
            Accum::Replace,
            v.as_ref(),
            h.transpose(),
            T::one(),
            par,
        );
        black_box(&ref_out);
    });
    let ta = median_time(|| {
        gemm(
            out.as_mut(),
            Accum::Replace,
            v.as_ref(),
            h.transpose(),
            T::one(),
            par,
        );
        black_box(&out);
    });
    let rel = max_rel_diff(ref_out.as_ref(), out.as_ref(), false);
    report(&format!("nmf V H^T  {tag} {n}x{m} k={k}"), tf, ta, rel);
}

/// Covariance-shaped `X^T X`: faer lower triangle against full gemm
///
/// ### Params
///
/// * `n` - Rows of X
/// * `p` - Columns of X
/// * `tag` - Float type label
fn bench_cov<T: Syrk>(n: usize, p: usize, tag: &str) {
    let x = random_mat::<T>(n, p, 4);
    let par = faer_parallelism();

    let mut ref_out = Mat::<T>::zeros(p, p);
    let mut out = Mat::<T>::zeros(p, p);
    let tf = median_time(|| {
        triangular_matmul(
            ref_out.as_mut(),
            BlockStructure::TriangularLower,
            Accum::Replace,
            x.transpose(),
            BlockStructure::Rectangular,
            x.as_ref(),
            BlockStructure::Rectangular,
            T::one(),
            par,
        );
        black_box(&ref_out);
    });
    let tff = median_time(|| {
        matmul(
            ref_out.as_mut(),
            Accum::Replace,
            x.transpose(),
            x.as_ref(),
            T::one(),
            par,
        );
        black_box(&ref_out);
    });
    let ta = median_time(|| {
        gemm(
            out.as_mut(),
            Accum::Replace,
            x.transpose(),
            x.as_ref(),
            T::one(),
            par,
        );
        black_box(&out);
    });
    let rel = max_rel_diff(ref_out.as_ref(), out.as_ref(), true);
    report(&format!("cov tri   {tag} {n}x{p}"), tf, ta, rel);
    report(&format!("cov full  {tag} {n}x{p}"), tff, ta, rel);

    let mut syrk_out = Mat::<T>::zeros(p, p);
    let ts = median_time(|| {
        T::syrk_xtx(x.as_ref(), &mut syrk_out);
        black_box(&syrk_out);
    });
    let rel = max_rel_diff(ref_out.as_ref(), syrk_out.as_ref(), true);
    report(&format!("cov syrk  {tag} {n}x{p} (vs tri)"), tf, ts, rel);
    report(&format!("cov syrk  {tag} {n}x{p} (vs accel gemm)"), ta, ts, rel);
}

/// `Par::Seq` tile GEMMs under rayon: `X_tile C^T`
///
/// ### Params
///
/// * `n` - Rows of X, split into `TILE_ROWS` tiles
/// * `d` - Columns of X and C
/// * `k` - Rows of C
/// * `tag` - Float type label
fn bench_tiled<T: BixverseFloat>(n: usize, d: usize, k: usize, tag: &str) {
    let x = random_mat::<T>(n, d, 5);
    let c = random_mat::<T>(k, d, 6);
    let starts: Vec<usize> = (0..n).step_by(TILE_ROWS).collect();

    let run = |use_gemm: bool| {
        starts.par_iter().for_each_init(
            || Mat::<T>::zeros(TILE_ROWS, k),
            |out, &s| {
                let rows = TILE_ROWS.min(n - s);
                let dst = out.as_mut().subrows_mut(0, rows);
                let lhs = x.as_ref().subrows(s, rows);
                if use_gemm {
                    gemm(dst, Accum::Replace, lhs, c.transpose(), T::one(), Par::Seq);
                } else {
                    matmul(dst, Accum::Replace, lhs, c.transpose(), T::one(), Par::Seq);
                }
                black_box(&out);
            },
        )
    };
    let tf = median_time(|| run(false));
    let ta = median_time(|| run(true));

    let tile = x.as_ref().subrows(0, TILE_ROWS.min(n));
    let mut ref_out = Mat::<T>::zeros(tile.nrows(), k);
    let mut out = Mat::<T>::zeros(tile.nrows(), k);
    matmul(
        ref_out.as_mut(),
        Accum::Replace,
        tile,
        c.transpose(),
        T::one(),
        Par::Seq,
    );
    gemm(
        out.as_mut(),
        Accum::Replace,
        tile,
        c.transpose(),
        T::one(),
        Par::Seq,
    );
    let rel = max_rel_diff(ref_out.as_ref(), out.as_ref(), false);
    report(&format!("tiled     {tag} {n}x{d} k={k}"), tf, ta, rel);
}

fn main() {
    let large = std::env::var("ACCEL_BENCH_LARGE").is_ok();
    let (n, m) = if large {
        (20_000, 2_000)
    } else {
        (5_000, 1_000)
    };
    let (cov_n, cov_p) = if large {
        (50_000, 2_000)
    } else {
        (5_000, 1_000)
    };
    let tiled_n = if large { 200_000 } else { 50_000 };

    println!(
        "VECLIB_MAXIMUM_THREADS={}",
        std::env::var("VECLIB_MAXIMUM_THREADS").unwrap_or_else(|_| "unset".into())
    );

    for k in [10, 50] {
        bench_nmf::<f32>(n, m, k, "f32");
        bench_nmf::<f64>(n, m, k, "f64");
    }
    bench_cov::<f32>(cov_n, cov_p, "f32");
    bench_cov::<f64>(cov_n, cov_p, "f64");
    for (d, k) in [(32, 64), (128, 256)] {
        bench_tiled::<f32>(tiled_n, d, k, "f32");
        bench_tiled::<f64>(tiled_n, d, k, "f64");
    }
}
