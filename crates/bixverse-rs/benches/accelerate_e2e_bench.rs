//! End-to-end wall clock of the functions the Accelerate GEMM routing touches.
//! Run once with the default features and once with `--no-default-features`
//! and compare: the binary itself does not know which backend it has. The
//! `accelerate` feature only changes anything on macOS.
//!
//! NMF runs a fixed number of iterations from a random init, so both builds do
//! the same work.
//!
//! Run with:
//! ```text
//! cargo bench --bench accelerate_e2e_bench
//! cargo bench --no-default-features --bench accelerate_e2e_bench
//! ```
//!
//! `ACCEL_BENCH_LARGE=1` switches to the larger shapes.

use std::hint::black_box;
use std::time::{Duration, Instant};

use faer::Mat;
use rand::prelude::*;
use rand::rngs::StdRng;

use bixverse_rs::core::base::cors_similarity::{column_pairwise_cor, column_pairwise_cov};
use bixverse_rs::methods::nmf_hals::dense::DenseInput;
use bixverse_rs::methods::nmf_hals::{HalsOpts, NmfInit, nmf_hals};
use bixverse_rs::prelude::*;

///////////////
// Constants //
///////////////

/// Timed repetitions per cell, after one warm-up. The median is reported.
const REPS: usize = 3;

/// Fixed HALS iteration count.
const NMF_ITERS: usize = 50;

/// NMF ranks swept.
const NMF_RANKS: [usize; 2] = [10, 50];

/// `(samples, features)` of the correlation cells, default shape.
const COR_SHAPE: (usize, usize) = (5_000, 1_000);

/// `(samples, features)` of the correlation cells under `ACCEL_BENCH_LARGE`.
const COR_SHAPE_LARGE: (usize, usize) = (50_000, 2_000);

/// `(rows, columns)` of V in the NMF cells, default shape.
const NMF_SHAPE: (usize, usize) = (5_000, 1_000);

/// `(rows, columns)` of V in the NMF cells under `ACCEL_BENCH_LARGE`.
const NMF_SHAPE_LARGE: (usize, usize) = (20_000, 2_000);

/////////////
// Helpers //
/////////////

/// Random matrix with entries in `[0, 1)`.
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
        T::from_f64(rng.random::<f64>()).expect("f64 fits T")
    })
}

/// Median wall clock of [`REPS`] runs after one warm-up.
///
/// ### Params
///
/// * `f` - Closure to time
///
/// ### Returns
///
/// Median duration.
fn median_of(mut f: impl FnMut()) -> Duration {
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

/////////////
// Benches //
/////////////

/// Correlation and covariance cells.
///
/// ### Params
///
/// * `n` - Rows (samples)
/// * `p` - Columns (features)
/// * `tag` - Float type label
fn bench_cor<T: BixverseFloat>(n: usize, p: usize, tag: &str) {
    let x = random_mat::<T>(n, p, 1);
    let t = median_of(|| {
        black_box(column_pairwise_cov(&x.as_ref()));
    });
    println!("column_pairwise_cov    {tag} {n}x{p}         {t:>9.2?}");
    let t = median_of(|| {
        black_box(column_pairwise_cor(&x.as_ref(), false));
    });
    println!("column_pairwise_cor    {tag} {n}x{p}         {t:>9.2?}");
}

/// Dense NMF HALS fit at fixed iterations.
///
/// ### Params
///
/// * `n` - Rows of V
/// * `m` - Columns of V
/// * `k` - Rank
/// * `tag` - Float type label
fn bench_nmf<T: BixverseFloat + Send + Sync>(n: usize, m: usize, k: usize, tag: &str) {
    let v = random_mat::<T>(n, m, 2);
    let input = DenseInput::new(v.as_ref()).expect("dense input");
    let opts = HalsOpts::new(
        NMF_ITERS,
        T::zero(),
        T::from_f64(1e-10).expect("f64 fits T"),
        NMF_ITERS + 1,
        NmfInit::Random { seed: 42 },
    );
    let t = median_of(|| {
        black_box(nmf_hals(&input, k, &opts, 0).expect("nmf"));
    });
    println!("nmf_hals {NMF_ITERS} iters     {tag} {n}x{m} k={k:<3}  {t:>9.2?}");
}

//////////
// Main //
//////////

fn main() {
    let large = std::env::var("ACCEL_BENCH_LARGE").is_ok();
    let (cn, cp) = if large { COR_SHAPE_LARGE } else { COR_SHAPE };
    let (nn, nm) = if large { NMF_SHAPE_LARGE } else { NMF_SHAPE };

    bench_cor::<f32>(cn, cp, "f32");
    bench_cor::<f64>(cn, cp, "f64");
    for k in NMF_RANKS {
        bench_nmf::<f32>(nn, nm, k, "f32");
        bench_nmf::<f64>(nn, nm, k, "f64");
    }
}
