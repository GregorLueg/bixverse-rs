//! End-to-end wall clock of the functions the Accelerate GEMM routing touches.
//! Run once with the default features and once with `--no-default-features`
//! and compare: the binary itself does not know which backend it has.
//!
//! NMF runs a fixed number of iterations from a random init, so both builds do
//! the same work.
//!
//! Run with:
//! ```
//! cargo bench --bench accelerate_e2e_bench
//! cargo bench --no-default-features --bench accelerate_e2e_bench
//! ```
//!
//! `ACCEL_BENCH_LARGE=1` switches to the larger shapes.

use std::hint::black_box;
use std::time::{Duration, Instant};

use bixverse_rs::core::base::cors_similarity::{column_pairwise_cor, column_pairwise_cov};
use bixverse_rs::methods::nmf_hals::dense::DenseInput;
use bixverse_rs::methods::nmf_hals::{HalsOpts, NmfInit, nmf_hals};
use bixverse_rs::prelude::*;
use faer::Mat;
use rand::prelude::*;
use rand::rngs::StdRng;

/// Timed repetitions per cell, after one warm-up.
const REPS: usize = 3;

/// Fixed HALS iteration count.
const NMF_ITERS: usize = 50;

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

/// Correlation and covariance cells
///
/// ### Params
///
/// * `n` - Rows (samples)
/// * `p` - Columns (features)
/// * `tag` - Float type label
fn bench_cor<T: BixverseFloat>(n: usize, p: usize, tag: &str) {
    let x = random_mat::<T>(n, p, 1);
    let t = median_time(|| {
        black_box(column_pairwise_cov(&x.as_ref()));
    });
    println!("column_pairwise_cov    {tag} {n}x{p}         {t:>9.2?}");
    let t = median_time(|| {
        black_box(column_pairwise_cor(&x.as_ref(), false));
    });
    println!("column_pairwise_cor    {tag} {n}x{p}         {t:>9.2?}");
}

/// Dense NMF HALS fit at fixed iterations
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
        T::from_f64(1e-10).unwrap(),
        NMF_ITERS + 1,
        NmfInit::Random { seed: 42 },
    );
    let t = median_time(|| {
        black_box(nmf_hals(&input, k, &opts, 0).expect("nmf"));
    });
    println!("nmf_hals {NMF_ITERS} iters     {tag} {n}x{m} k={k:<3}  {t:>9.2?}");
}

fn main() {
    let large = std::env::var("ACCEL_BENCH_LARGE").is_ok();
    let (cn, cp) = if large {
        (50_000, 2_000)
    } else {
        (5_000, 1_000)
    };
    let (nn, nm) = if large {
        (20_000, 2_000)
    } else {
        (5_000, 1_000)
    };

    bench_cor::<f32>(cn, cp, "f32");
    bench_cor::<f64>(cn, cp, "f64");
    for k in [10, 50] {
        bench_nmf::<f32>(nn, nm, k, "f32");
        bench_nmf::<f64>(nn, nm, k, "f64");
    }
}
