//! Targeted SIMD implementations to accelerate specific hot loops in bixverse
//!
//! The 128-bit reduction arms go through `wide`, which lowers to SSE2 on
//! x86_64 and NEON on aarch64. Both are baseline for their architecture, so
//! both are real in a stock build, and `wide` keeps them to one code path.
//!
//! The 256- and 512-bit arms are hand-written `std::arch::x86_64` intrinsics
//! behind `#[target_feature]`. `wide`'s wider types cannot be used here:
//! `f32x8` is only backed by `__m256` when the *whole crate* is compiled with
//! `target_feature = "avx"`, which no stock build and no rextendr build does,
//! so it would silently degrade to two SSE2 halves. `#[target_feature]`
//! compiles the body unconditionally and is safe to enter behind
//! `is_x86_feature_detected!`, which is what `detect_simd_level` provides.
//!
//! The fused argmin is the exception: it needs compare-and-blend, which is the
//! part of `wide`'s surface that has churned across minor releases (`CmpLt`
//! deprecated in 1.5, `blend` in 1.6), and pinning that down across the range
//! of versions a downstream lockfile might resolve is not worth it for one
//! kernel. All of its arms, including 128-bit, are written per-architecture
//! instead. The reductions only need arithmetic and `reduce_add`, which are
//! stable, so they stay on `wide`.
//!
//! Every hand-written arm follows the same shape, so the two function-local
//! constants mean the same thing throughout: `W` is the lane count forced by
//! the intrinsic width (4, 8 or 16 for `f32`; 2, 4 or 8 for `f64`) and
//! `BLOCK = W * UNROLL` is what one pass of the unrolled loop consumes. Each
//! arm runs the block loop, folds its accumulators, sweeps any remaining whole
//! vectors, then finishes the last `< W` elements scalar.

use std::sync::OnceLock;
use wide::{f32x4, f64x2};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Independent accumulators carried through the vectorised loops.
///
/// Both `add` and `fma` have roughly four cycles of latency against a
/// two-per-cycle issue rate, and the argmin's compare-and-blend chain has the
/// same shape. A single accumulator serialises on that latency and leaves most
/// of the issue width idle regardless of vector width. Four covers it; more
/// only lengthens the final cross-accumulator fold.
///
/// Indicative, from an A/B of `argmin_diff_simd_f32` on one aarch64 machine:
/// 1.4x at `len` 200 rising to 2.9x by `len` 16384, the gain growing with
/// length because the fixed fold cost amortises. Retune by flipping this
/// constant and timing the same call, not by reasoning about the number.
pub(crate) const UNROLL: usize = 4;

/// Enum for the different architectures and potential SIMD levels
#[derive(Clone, Copy, Debug)]
pub enum SimdLevel {
    /// Scalar version
    Scalar,
    /// 128-bit (also covers NEON which is used by Apple)
    Sse,
    /// 256-bit
    Avx2,
    /// 512-bit
    Avx512,
}

/// SIMD level static generated once
static SIMD_LEVEL: OnceLock<SimdLevel> = OnceLock::new();

/// Function to detect which SIMD implementation to use
///
/// The AVX-2 probe also requires FMA, because every 256-bit arm here is
/// written with `_mm256_fmadd_ps` and is entered under
/// `#[target_feature(enable = "avx2", enable = "fma")]`. No shipping x86 part
/// has AVX-2 without FMA, so this costs nothing in practice.
///
/// ### Returns
///
/// The widest SIMD level this CPU supports.
pub fn detect_simd_level() -> SimdLevel {
    *SIMD_LEVEL.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return SimdLevel::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return SimdLevel::Avx2;
            }
            if is_x86_feature_detected!("sse4.1") {
                return SimdLevel::Sse;
            }
            return SimdLevel::Scalar;
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on aarch64
            SimdLevel::Sse
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            SimdLevel::Scalar
        }
    })
}

/////////////////
// x86 helpers //
/////////////////

/// Horizontal sum of a 256-bit f32 vector.
///
/// Spilling to the stack rather than shuffling: this runs once per call, so
/// the shuffle sequence would buy nothing measurable and costs readability.
///
/// ### Params
///
/// * `v` - The vector to reduce.
///
/// ### Returns
///
/// The sum of the eight lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn hsum_avx_f32(v: __m256) -> f32 {
    unsafe {
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), v);
        tmp.iter().sum()
    }
}

/// Horizontal sum of a 256-bit f64 vector.
///
/// ### Params
///
/// * `v` - The vector to reduce.
///
/// ### Returns
///
/// The sum of the four lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn hsum_avx_f64(v: __m256d) -> f64 {
    unsafe {
        let mut tmp = [0.0f64; 4];
        _mm256_storeu_pd(tmp.as_mut_ptr(), v);
        tmp.iter().sum()
    }
}

//////////////////////////////////
// f32-specific implementations //
//////////////////////////////////

/////////////////
// Sum squares //
/////////////////

/// SIMD squared sum of a slice of f32 (scalar)
///
/// ### Params
///
/// * `a` - The slice of f32 values to sum.
///
/// ### Returns
///
/// Squared sum
#[inline(always)]
fn sum_squares_scalar_f32(a: &[f32]) -> f32 {
    a.iter().map(|&x| x * x).sum()
}

/// SIMD squared sum of a slice of f32 (128-bit)
///
/// ### Params
///
/// * `a` - The slice of f32 values to sum.
///
/// ### Returns
///
/// Squared sum
#[inline(always)]
fn sum_squares_sse_f32(a: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = f32x4::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        for i in 0..chunks {
            let va = f32x4::from(*(a_ptr.add(i * 4) as *const [f32; 4]));
            acc += va * va;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        sum += a[i] * a[i];
    }
    sum
}

/// SIMD squared sum of a slice of f32 (256-bit)
///
/// ### Params
///
/// * `a` - The slice of f32 values to sum.
///
/// ### Returns
///
/// Squared sum
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sum_squares_avx2_f32(a: &[f32]) -> f32 {
    const W: usize = 8;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let mut acc = [_mm256_setzero_ps(); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                let va = _mm256_loadu_ps(a_ptr.add(base + u * W));
                *acc = _mm256_fmadd_ps(va, va, *acc);
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = _mm256_add_ps(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let va = _mm256_loadu_ps(a_ptr.add(i));
            total = _mm256_fmadd_ps(va, va, total);
            i += W;
        }

        let mut sum = hsum_avx_f32(total);
        while i < len {
            sum += a[i] * a[i];
            i += 1;
        }
        sum
    }
}

/// SIMD squared sum of a slice of f32 (512-bit)
///
/// ### Params
///
/// * `a` - The slice of f32 values to sum.
///
/// ### Returns
///
/// Squared sum
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_squares_avx512_f32(a: &[f32]) -> f32 {
    const W: usize = 16;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let mut acc = [_mm512_setzero_ps(); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                let va = _mm512_loadu_ps(a_ptr.add(base + u * W));
                *acc = _mm512_fmadd_ps(va, va, *acc);
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = _mm512_add_ps(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let va = _mm512_loadu_ps(a_ptr.add(i));
            total = _mm512_fmadd_ps(va, va, total);
            i += W;
        }

        let mut sum = _mm512_reduce_add_ps(total);
        while i < len {
            sum += a[i] * a[i];
            i += 1;
        }
        sum
    }
}

/// SIMD squared sum of a slice of f32 (dispatch)
///
/// ### Params
///
/// * `a` - The slice of f32 values to sum.
///
/// ### Returns
///
/// Squared sum
#[inline]
pub fn sum_squares_simd_f32(a: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: each arm is entered only once `detect_simd_level` has confirmed
    // the CPU reports the features that arm is compiled for.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Avx512 => sum_squares_avx512_f32(a),
            SimdLevel::Avx2 => sum_squares_avx2_f32(a),
            SimdLevel::Sse => sum_squares_sse_f32(a),
            SimdLevel::Scalar => sum_squares_scalar_f32(a),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    match detect_simd_level() {
        SimdLevel::Sse => sum_squares_sse_f32(a),
        _ => sum_squares_scalar_f32(a),
    }
}

//////////////////////
// General versions //
//////////////////////

/// Trait for SIMD-accelerated dot product, replacing the ann-search-rs
/// SimdDistance dependency.
pub trait BixverseSimd:
    Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = Self>
{
    /// Compute the dot product of two slices using SIMD where available.
    ///
    /// ### Params
    ///
    /// * `a` - First slice
    /// * `b` - Second slice (must be the same length as `a`)
    ///
    /// ### Returns
    ///
    /// The dot product of `a` and `b`
    fn bxv_dot_simd(a: &[Self], b: &[Self]) -> Self;

    /// Compute the sum over a slice with SIMD
    ///
    /// ### Params
    ///
    /// * `a` - Slice for which to calculate the sum
    ///
    /// ### Returns
    ///
    /// The sum of the vector
    fn bxv_sum(x: &[Self]) -> Self;

    /// Sum of squared deviations from the mean
    ///
    /// ### Params
    ///
    /// * `x` - Vector for which to calculate the squared deviations from the
    ///   mean.
    /// * `mean` - The mean.
    ///
    /// ### Returns
    ///
    /// The sum of the squared deviations.
    fn bxv_sum_squared_deviation(x: &[Self], mean: Self) -> Self;
}

impl BixverseSimd for f32 {
    #[inline]
    fn bxv_dot_simd(a: &[f32], b: &[f32]) -> f32 {
        dot_simd_f32(a, b)
    }

    #[inline]
    fn bxv_sum(x: &[f32]) -> f32 {
        sum_simd_f32(x)
    }

    #[inline]
    fn bxv_sum_squared_deviation(x: &[f32], mean: f32) -> f32 {
        sum_squared_dev_simd_f32(x, mean)
    }
}

impl BixverseSimd for f64 {
    #[inline]
    fn bxv_dot_simd(a: &[f64], b: &[f64]) -> f64 {
        dot_simd_f64(a, b)
    }

    #[inline]
    fn bxv_sum(x: &[f64]) -> f64 {
        sum_simd_f64(x)
    }

    #[inline]
    fn bxv_sum_squared_deviation(x: &[f64], mean: f64) -> f64 {
        sum_squared_dev_simd_f64(x, mean)
    }
}

//////////////////
// Dot products //
//////////////////

/////////
// f32 //
/////////

/// SIMD dot product of two slices of f32 (scalar)
///
/// ### Params
///
/// * `a` - The first slice of f32 values.
/// * `b` - The second slice of f32 values.
///
/// ### Returns
///
/// Dot product
#[inline(always)]
fn dot_scalar_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// SIMD dot product of two slices of f32 (128-bit)
///
/// ### Params
///
/// * `a` - The first slice of f32 values.
/// * `b` - The second slice of f32 values.
///
/// ### Returns
///
/// Dot product
#[inline(always)]
fn dot_sse_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = f32x4::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        for i in 0..chunks {
            let va = f32x4::from(*(a_ptr.add(i * 4) as *const [f32; 4]));
            let vb = f32x4::from(*(b_ptr.add(i * 4) as *const [f32; 4]));
            acc += va * vb;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        sum += a[i] * b[i];
    }
    sum
}

/// SIMD dot product of two slices of f32 (256-bit)
///
/// ### Params
///
/// * `a` - The first slice of f32 values.
/// * `b` - The second slice of f32 values.
///
/// ### Returns
///
/// Dot product
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2_f32(a: &[f32], b: &[f32]) -> f32 {
    const W: usize = 8;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let mut acc = [_mm256_setzero_ps(); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                let off = base + u * W;
                let va = _mm256_loadu_ps(a_ptr.add(off));
                let vb = _mm256_loadu_ps(b_ptr.add(off));
                *acc = _mm256_fmadd_ps(va, vb, *acc);
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = _mm256_add_ps(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let va = _mm256_loadu_ps(a_ptr.add(i));
            let vb = _mm256_loadu_ps(b_ptr.add(i));
            total = _mm256_fmadd_ps(va, vb, total);
            i += W;
        }

        let mut sum = hsum_avx_f32(total);
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }
}

/// SIMD dot product of two slices of f32 (512-bit)
///
/// ### Params
///
/// * `a` - The first slice of f32 values.
/// * `b` - The second slice of f32 values.
///
/// ### Returns
///
/// Dot product
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512_f32(a: &[f32], b: &[f32]) -> f32 {
    const W: usize = 16;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let mut acc = [_mm512_setzero_ps(); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                let off = base + u * W;
                let va = _mm512_loadu_ps(a_ptr.add(off));
                let vb = _mm512_loadu_ps(b_ptr.add(off));
                *acc = _mm512_fmadd_ps(va, vb, *acc);
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = _mm512_add_ps(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let va = _mm512_loadu_ps(a_ptr.add(i));
            let vb = _mm512_loadu_ps(b_ptr.add(i));
            total = _mm512_fmadd_ps(va, vb, total);
            i += W;
        }

        let mut sum = _mm512_reduce_add_ps(total);
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }
}

/// SIMD dot product of two slices of f32 (dispatch)
///
/// Dispatches to the best available SIMD implementation at runtime.
///
/// ### Params
///
/// * `a` - The first slice of f32 values.
/// * `b` - The second slice of f32 values.
///
/// ### Returns
///
/// Dot product
#[inline]
pub fn dot_simd_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: see `sum_squares_simd_f32`.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Avx512 => dot_avx512_f32(a, b),
            SimdLevel::Avx2 => dot_avx2_f32(a, b),
            SimdLevel::Sse => dot_sse_f32(a, b),
            SimdLevel::Scalar => dot_scalar_f32(a, b),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    match detect_simd_level() {
        SimdLevel::Sse => dot_sse_f32(a, b),
        _ => dot_scalar_f32(a, b),
    }
}

/////////
// f64 //
/////////

/// SIMD dot product of two slices of f64 (scalar)
///
/// ### Params
///
/// * `a` - The first slice of f64 values.
/// * `b` - The second slice of f64 values.
///
/// ### Returns
///
/// Dot product
#[inline(always)]
fn dot_scalar_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// SIMD dot product of two slices of f64 (128-bit)
///
/// ### Params
///
/// * `a` - The first slice of f64 values.
/// * `b` - The second slice of f64 values.
///
/// ### Returns
///
/// Dot product
#[inline(always)]
fn dot_sse_f64(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let chunks = len / 2;
    let mut acc = f64x2::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        for i in 0..chunks {
            let va = f64x2::from(*(a_ptr.add(i * 2) as *const [f64; 2]));
            let vb = f64x2::from(*(b_ptr.add(i * 2) as *const [f64; 2]));
            acc += va * vb;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 2)..len {
        sum += a[i] * b[i];
    }
    sum
}

/// SIMD dot product of two slices of f64 (256-bit)
///
/// ### Params
///
/// * `a` - The first slice of f64 values.
/// * `b` - The second slice of f64 values.
///
/// ### Returns
///
/// Dot product
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2_f64(a: &[f64], b: &[f64]) -> f64 {
    const W: usize = 4;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let mut acc = [_mm256_setzero_pd(); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                let off = base + u * W;
                let va = _mm256_loadu_pd(a_ptr.add(off));
                let vb = _mm256_loadu_pd(b_ptr.add(off));
                *acc = _mm256_fmadd_pd(va, vb, *acc);
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = _mm256_add_pd(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let va = _mm256_loadu_pd(a_ptr.add(i));
            let vb = _mm256_loadu_pd(b_ptr.add(i));
            total = _mm256_fmadd_pd(va, vb, total);
            i += W;
        }

        let mut sum = hsum_avx_f64(total);
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }
}

/// SIMD dot product of two slices of f64 (512-bit)
///
/// ### Params
///
/// * `a` - The first slice of f64 values.
/// * `b` - The second slice of f64 values.
///
/// ### Returns
///
/// Dot product
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512_f64(a: &[f64], b: &[f64]) -> f64 {
    const W: usize = 8;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let mut acc = [_mm512_setzero_pd(); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                let off = base + u * W;
                let va = _mm512_loadu_pd(a_ptr.add(off));
                let vb = _mm512_loadu_pd(b_ptr.add(off));
                *acc = _mm512_fmadd_pd(va, vb, *acc);
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = _mm512_add_pd(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let va = _mm512_loadu_pd(a_ptr.add(i));
            let vb = _mm512_loadu_pd(b_ptr.add(i));
            total = _mm512_fmadd_pd(va, vb, total);
            i += W;
        }

        let mut sum = _mm512_reduce_add_pd(total);
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }
}

/// SIMD dot product of two slices of f64 (dispatch)
///
/// Dispatches to the best available SIMD implementation at runtime.
///
/// ### Params
///
/// * `a` - The first slice of f64 values.
/// * `b` - The second slice of f64 values.
///
/// ### Returns
///
/// Dot product
#[inline]
pub fn dot_simd_f64(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: see `sum_squares_simd_f32`.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Avx512 => dot_avx512_f64(a, b),
            SimdLevel::Avx2 => dot_avx2_f64(a, b),
            SimdLevel::Sse => dot_sse_f64(a, b),
            SimdLevel::Scalar => dot_scalar_f64(a, b),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    match detect_simd_level() {
        SimdLevel::Sse => dot_sse_f64(a, b),
        _ => dot_scalar_f64(a, b),
    }
}

///////////////////////////
// Fused subtract-argmin //
///////////////////////////

/// Fused subtract-and-argmin over two slices of f32 (scalar)
///
/// ### Params
///
/// * `a` - The minuend slice.
/// * `b` - The subtrahend slice, same length as `a`.
///
/// ### Returns
///
/// `(index, value)` of the smallest `a[i] - b[i]`, ties to the lowest index.
#[inline(always)]
fn argmin_diff_scalar_f32(a: &[f32], b: &[f32]) -> (usize, f32) {
    let mut min_val = a[0] - b[0];
    let mut min_idx = 0usize;
    for i in 1..a.len() {
        let diff = a[i] - b[i];
        if diff < min_val {
            min_val = diff;
            min_idx = i;
        }
    }
    (min_idx, min_val)
}

/// Reduce per-lane running minima to a single `(index, value)`.
///
/// Lane indices are carried as `f32`, which is exact for any index below 2^24
/// and avoids a bitcast between the float compare mask and an integer vector.
///
/// ### Params
///
/// * `vals` - Per-lane minimum values.
/// * `idxs` - Per-lane indices at which those minima occurred.
///
/// ### Returns
///
/// `(index, value)` of the smallest entry, ties to the lowest index.
#[inline(always)]
fn reduce_lane_argmin_f32(vals: &[f32], idxs: &[f32]) -> (usize, f32) {
    let mut min_val = f32::INFINITY;
    let mut min_idx = usize::MAX;
    for (&value, &index) in vals.iter().zip(idxs.iter()) {
        let index = index as usize;
        if value < min_val || (value == min_val && index < min_idx) {
            min_val = value;
            min_idx = index;
        }
    }
    (min_idx, min_val)
}

/// Merge one accumulator pair into another, lane by lane (128-bit, SSE4.1).
///
/// The tie-break has to compare indices explicitly rather than rely on
/// accumulator order. Within one accumulator indices grow monotonically, so a
/// strict `<` in the scan keeps the earliest. Across accumulators that no
/// longer holds: accumulator 1's minimum may come from an earlier block than
/// accumulator 0's, so its index can be the lower of the two.
///
/// ### Params
///
/// * `best` - Running per-lane minima.
/// * `best_idx` - Indices of those minima.
/// * `cand` - Candidate per-lane minima to merge in.
/// * `cand_idx` - Indices of the candidates.
///
/// ### Returns
///
/// The merged `(values, indices)` pair, ties to the lower index.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn fold_argmin_sse(
    best: __m128,
    best_idx: __m128,
    cand: __m128,
    cand_idx: __m128,
) -> (__m128, __m128) {
    let lt = _mm_cmplt_ps(cand, best);
    let eq = _mm_cmpeq_ps(cand, best);
    let ilt = _mm_cmplt_ps(cand_idx, best_idx);
    let take = _mm_or_ps(lt, _mm_and_ps(eq, ilt));
    (
        _mm_blendv_ps(best, cand, take),
        _mm_blendv_ps(best_idx, cand_idx, take),
    )
}

/// Fused subtract-and-argmin over two slices of f32 (128-bit, SSE4.1)
///
/// ### Params
///
/// * `a` - The minuend slice.
/// * `b` - The subtrahend slice, same length as `a`.
///
/// ### Returns
///
/// `(index, value)` of the smallest `a[i] - b[i]`, ties to the lowest index.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn argmin_diff_sse_f32(a: &[f32], b: &[f32]) -> (usize, f32) {
    const W: usize = 4;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let n_blocks = len / BLOCK;

        let mut best = [_mm_set1_ps(f32::INFINITY); UNROLL];
        let mut best_idx = [_mm_setzero_ps(); UNROLL];
        let lane_base = _mm_setr_ps(0.0, 1.0, 2.0, 3.0);
        let mut lane: [__m128; UNROLL] =
            std::array::from_fn(|u| _mm_add_ps(lane_base, _mm_set1_ps((u * W) as f32)));
        let step = _mm_set1_ps(BLOCK as f32);

        for i in 0..n_blocks {
            let base = i * BLOCK;
            for u in 0..UNROLL {
                let off = base + u * W;
                let diff = _mm_sub_ps(_mm_loadu_ps(a_ptr.add(off)), _mm_loadu_ps(b_ptr.add(off)));
                // Strict `<` keeps the earliest block within a lane; ties
                // between accumulators are resolved by `fold_argmin_sse`.
                let mask = _mm_cmplt_ps(diff, best[u]);
                best[u] = _mm_blendv_ps(best[u], diff, mask);
                best_idx[u] = _mm_blendv_ps(best_idx[u], lane[u], mask);
                lane[u] = _mm_add_ps(lane[u], step);
            }
        }

        let mut acc = best[0];
        let mut acc_idx = best_idx[0];
        for u in 1..UNROLL {
            (acc, acc_idx) = fold_argmin_sse(acc, acc_idx, best[u], best_idx[u]);
        }

        // Whole vectors left over by the block loop. Their indices are strictly
        // greater than anything already in `acc`, so a strict `<` still keeps
        // the earliest on a tie.
        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let diff = _mm_sub_ps(_mm_loadu_ps(a_ptr.add(i)), _mm_loadu_ps(b_ptr.add(i)));
            let lane_v = _mm_add_ps(lane_base, _mm_set1_ps(i as f32));
            let mask = _mm_cmplt_ps(diff, acc);
            acc = _mm_blendv_ps(acc, diff, mask);
            acc_idx = _mm_blendv_ps(acc_idx, lane_v, mask);
            i += W;
        }

        let mut vals = [0.0f32; W];
        let mut idxs = [0.0f32; W];
        _mm_storeu_ps(vals.as_mut_ptr(), acc);
        _mm_storeu_ps(idxs.as_mut_ptr(), acc_idx);
        let (mut min_idx, mut min_val) = reduce_lane_argmin_f32(&vals, &idxs);

        while i < len {
            let diff = a[i] - b[i];
            if diff < min_val {
                min_val = diff;
                min_idx = i;
            }
            i += 1;
        }

        (min_idx, min_val)
    }
}

/// Merge one accumulator pair into another, lane by lane (NEON).
///
/// See `fold_argmin_sse` for why the index comparison is needed.
///
/// ### Params
///
/// * `best` - Running per-lane minima.
/// * `best_idx` - Indices of those minima.
/// * `cand` - Candidate per-lane minima to merge in.
/// * `cand_idx` - Indices of the candidates.
///
/// ### Returns
///
/// The merged `(values, indices)` pair, ties to the lower index.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fold_argmin_neon(
    best: float32x4_t,
    best_idx: float32x4_t,
    cand: float32x4_t,
    cand_idx: float32x4_t,
) -> (float32x4_t, float32x4_t) {
    let lt = vcltq_f32(cand, best);
    let eq = vceqq_f32(cand, best);
    let ilt = vcltq_f32(cand_idx, best_idx);
    let take = vorrq_u32(lt, vandq_u32(eq, ilt));
    (
        vbslq_f32(take, cand, best),
        vbslq_f32(take, cand_idx, best_idx),
    )
}

/// Fused subtract-and-argmin over two slices of f32 (128-bit, NEON)
///
/// ### Params
///
/// * `a` - The minuend slice.
/// * `b` - The subtrahend slice, same length as `a`.
///
/// ### Returns
///
/// `(index, value)` of the smallest `a[i] - b[i]`, ties to the lowest index.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn argmin_diff_neon_f32(a: &[f32], b: &[f32]) -> (usize, f32) {
    const W: usize = 4;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let n_blocks = len / BLOCK;

        let mut best = [vdupq_n_f32(f32::INFINITY); UNROLL];
        let mut best_idx = [vdupq_n_f32(0.0); UNROLL];
        let lane_seed: [f32; W] = [0.0, 1.0, 2.0, 3.0];
        let lane_base = vld1q_f32(lane_seed.as_ptr());
        let mut lane: [float32x4_t; UNROLL] =
            std::array::from_fn(|u| vaddq_f32(lane_base, vdupq_n_f32((u * W) as f32)));
        let step = vdupq_n_f32(BLOCK as f32);

        for i in 0..n_blocks {
            let base = i * BLOCK;
            for u in 0..UNROLL {
                let off = base + u * W;
                let diff = vsubq_f32(vld1q_f32(a_ptr.add(off)), vld1q_f32(b_ptr.add(off)));
                // Strict `<` keeps the earliest block within a lane; ties
                // between accumulators are resolved by `fold_argmin_neon`.
                let mask = vcltq_f32(diff, best[u]);
                best[u] = vbslq_f32(mask, diff, best[u]);
                best_idx[u] = vbslq_f32(mask, lane[u], best_idx[u]);
                lane[u] = vaddq_f32(lane[u], step);
            }
        }

        let mut acc = best[0];
        let mut acc_idx = best_idx[0];
        for u in 1..UNROLL {
            (acc, acc_idx) = fold_argmin_neon(acc, acc_idx, best[u], best_idx[u]);
        }

        // See the SSE arm: leftover whole vectors carry strictly higher
        // indices, so the strict `<` is still the right tie-break.
        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let diff = vsubq_f32(vld1q_f32(a_ptr.add(i)), vld1q_f32(b_ptr.add(i)));
            let lane_v = vaddq_f32(lane_base, vdupq_n_f32(i as f32));
            let mask = vcltq_f32(diff, acc);
            acc = vbslq_f32(mask, diff, acc);
            acc_idx = vbslq_f32(mask, lane_v, acc_idx);
            i += W;
        }

        let mut vals = [0.0f32; W];
        let mut idxs = [0.0f32; W];
        vst1q_f32(vals.as_mut_ptr(), acc);
        vst1q_f32(idxs.as_mut_ptr(), acc_idx);
        let (mut min_idx, mut min_val) = reduce_lane_argmin_f32(&vals, &idxs);

        while i < len {
            let diff = a[i] - b[i];
            if diff < min_val {
                min_val = diff;
                min_idx = i;
            }
            i += 1;
        }

        (min_idx, min_val)
    }
}

/// Merge one accumulator pair into another, lane by lane (256-bit).
///
/// See `fold_argmin_sse` for why the index comparison is needed.
///
/// ### Params
///
/// * `best` - Running per-lane minima.
/// * `best_idx` - Indices of those minima.
/// * `cand` - Candidate per-lane minima to merge in.
/// * `cand_idx` - Indices of the candidates.
///
/// ### Returns
///
/// The merged `(values, indices)` pair, ties to the lower index.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn fold_argmin_avx(
    best: __m256,
    best_idx: __m256,
    cand: __m256,
    cand_idx: __m256,
) -> (__m256, __m256) {
    let lt = _mm256_cmp_ps(cand, best, _CMP_LT_OQ);
    let eq = _mm256_cmp_ps(cand, best, _CMP_EQ_OQ);
    let ilt = _mm256_cmp_ps(cand_idx, best_idx, _CMP_LT_OQ);
    let take = _mm256_or_ps(lt, _mm256_and_ps(eq, ilt));
    (
        _mm256_blendv_ps(best, cand, take),
        _mm256_blendv_ps(best_idx, cand_idx, take),
    )
}

/// Fused subtract-and-argmin over two slices of f32 (256-bit)
///
/// ### Params
///
/// * `a` - The minuend slice.
/// * `b` - The subtrahend slice, same length as `a`.
///
/// ### Returns
///
/// `(index, value)` of the smallest `a[i] - b[i]`, ties to the lowest index.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn argmin_diff_avx2_f32(a: &[f32], b: &[f32]) -> (usize, f32) {
    const W: usize = 8;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let n_blocks = len / BLOCK;

        let mut best = [_mm256_set1_ps(f32::INFINITY); UNROLL];
        let mut best_idx = [_mm256_setzero_ps(); UNROLL];
        let lane_base = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
        let mut lane: [__m256; UNROLL] =
            std::array::from_fn(|u| _mm256_add_ps(lane_base, _mm256_set1_ps((u * W) as f32)));
        let step = _mm256_set1_ps(BLOCK as f32);

        for i in 0..n_blocks {
            let base = i * BLOCK;
            for u in 0..UNROLL {
                let off = base + u * W;
                let va = _mm256_loadu_ps(a_ptr.add(off));
                let vb = _mm256_loadu_ps(b_ptr.add(off));
                let diff = _mm256_sub_ps(va, vb);
                // Strict `<` keeps the earliest block within a lane; ties
                // between accumulators are resolved by `fold_argmin_avx`.
                let mask = _mm256_cmp_ps(diff, best[u], _CMP_LT_OQ);
                best[u] = _mm256_blendv_ps(best[u], diff, mask);
                best_idx[u] = _mm256_blendv_ps(best_idx[u], lane[u], mask);
                lane[u] = _mm256_add_ps(lane[u], step);
            }
        }

        let mut acc = best[0];
        let mut acc_idx = best_idx[0];
        for u in 1..UNROLL {
            (acc, acc_idx) = fold_argmin_avx(acc, acc_idx, best[u], best_idx[u]);
        }

        // See the SSE arm: leftover whole vectors carry strictly higher
        // indices, so the strict `<` is still the right tie-break.
        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let diff = _mm256_sub_ps(_mm256_loadu_ps(a_ptr.add(i)), _mm256_loadu_ps(b_ptr.add(i)));
            let lane_v = _mm256_add_ps(lane_base, _mm256_set1_ps(i as f32));
            let mask = _mm256_cmp_ps(diff, acc, _CMP_LT_OQ);
            acc = _mm256_blendv_ps(acc, diff, mask);
            acc_idx = _mm256_blendv_ps(acc_idx, lane_v, mask);
            i += W;
        }

        let mut vals = [0.0f32; W];
        let mut idxs = [0.0f32; W];
        _mm256_storeu_ps(vals.as_mut_ptr(), acc);
        _mm256_storeu_ps(idxs.as_mut_ptr(), acc_idx);
        let (mut min_idx, mut min_val) = reduce_lane_argmin_f32(&vals, &idxs);

        while i < len {
            let diff = a[i] - b[i];
            if diff < min_val {
                min_val = diff;
                min_idx = i;
            }
            i += 1;
        }

        (min_idx, min_val)
    }
}

/// Fused subtract-and-argmin over two slices of f32 (512-bit)
///
/// Uses mask registers rather than a blend vector, so the compare produces a
/// `__mmask16` directly and the tie-break folds with plain integer bit ops.
///
/// ### Params
///
/// * `a` - The minuend slice.
/// * `b` - The subtrahend slice, same length as `a`.
///
/// ### Returns
///
/// `(index, value)` of the smallest `a[i] - b[i]`, ties to the lowest index.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn argmin_diff_avx512_f32(a: &[f32], b: &[f32]) -> (usize, f32) {
    const W: usize = 16;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let n_blocks = len / BLOCK;

        let mut best = [_mm512_set1_ps(f32::INFINITY); UNROLL];
        let mut best_idx = [_mm512_setzero_ps(); UNROLL];
        let lane_seed: [f32; W] = std::array::from_fn(|i| i as f32);
        let lane_base = _mm512_loadu_ps(lane_seed.as_ptr());
        let mut lane: [__m512; UNROLL] =
            std::array::from_fn(|u| _mm512_add_ps(lane_base, _mm512_set1_ps((u * W) as f32)));
        let step = _mm512_set1_ps(BLOCK as f32);

        for i in 0..n_blocks {
            let base = i * BLOCK;
            for u in 0..UNROLL {
                let off = base + u * W;
                let va = _mm512_loadu_ps(a_ptr.add(off));
                let vb = _mm512_loadu_ps(b_ptr.add(off));
                let diff = _mm512_sub_ps(va, vb);
                // Strict `<` keeps the earliest block within a lane; ties
                // between accumulators are resolved in the fold below.
                let mask = _mm512_cmp_ps_mask(diff, best[u], _CMP_LT_OQ);
                best[u] = _mm512_mask_blend_ps(mask, best[u], diff);
                best_idx[u] = _mm512_mask_blend_ps(mask, best_idx[u], lane[u]);
                lane[u] = _mm512_add_ps(lane[u], step);
            }
        }

        let mut acc = best[0];
        let mut acc_idx = best_idx[0];
        for u in 1..UNROLL {
            let lt = _mm512_cmp_ps_mask(best[u], acc, _CMP_LT_OQ);
            let eq = _mm512_cmp_ps_mask(best[u], acc, _CMP_EQ_OQ);
            let ilt = _mm512_cmp_ps_mask(best_idx[u], acc_idx, _CMP_LT_OQ);
            let take = lt | (eq & ilt);
            acc = _mm512_mask_blend_ps(take, acc, best[u]);
            acc_idx = _mm512_mask_blend_ps(take, acc_idx, best_idx[u]);
        }

        // See the SSE arm: leftover whole vectors carry strictly higher
        // indices, so the strict `<` is still the right tie-break.
        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let diff = _mm512_sub_ps(_mm512_loadu_ps(a_ptr.add(i)), _mm512_loadu_ps(b_ptr.add(i)));
            let lane_v = _mm512_add_ps(lane_base, _mm512_set1_ps(i as f32));
            let mask = _mm512_cmp_ps_mask(diff, acc, _CMP_LT_OQ);
            acc = _mm512_mask_blend_ps(mask, acc, diff);
            acc_idx = _mm512_mask_blend_ps(mask, acc_idx, lane_v);
            i += W;
        }

        let mut vals = [0.0f32; W];
        let mut idxs = [0.0f32; W];
        _mm512_storeu_ps(vals.as_mut_ptr(), acc);
        _mm512_storeu_ps(idxs.as_mut_ptr(), acc_idx);
        let (mut min_idx, mut min_val) = reduce_lane_argmin_f32(&vals, &idxs);

        while i < len {
            let diff = a[i] - b[i];
            if diff < min_val {
                min_val = diff;
                min_idx = i;
            }
            i += 1;
        }

        (min_idx, min_val)
    }
}

/// Fused subtract-and-argmin over two slices of f32 (dispatch)
///
/// Computes `argmin_i (a[i] - b[i])` without materialising the difference. The
/// scalar form of this loop does not auto-vectorise, because the running
/// minimum is a data-dependent branch, so it is the one pass in the Frank-Wolfe
/// column update that stays scalar unless it is written out by hand.
///
/// The compare-and-blend chain is loop-carried, so each vector arm carries
/// `UNROLL` independent accumulator pairs, folds them once the block loop is
/// done, sweeps any leftover whole vectors through the merged pair, and only
/// then falls back to scalar for the last `< W` elements. Leaving the whole
/// `BLOCK` remainder to the scalar tail costs more than the unroll wins at
/// small `k`, which is why the extra pass is there.
///
/// Two caveats, neither reachable from the Frank-Wolfe callers but both visible
/// to anyone else calling this:
///
/// - Lane indices are carried as `f32`, exact only below `2^24`. Longer slices
///   would return a rounded index.
/// - Non-finite differences are not handled consistently across the arms. The
///   scalar path seeds from `a[0] - b[0]`, so a `NaN` there poisons every later
///   comparison and it returns index 0; the vector paths seed at `+INFINITY`
///   and never blend a `NaN` in, so they return the minimum over the finite
///   differences. A lane whose only candidates are `+INFINITY` keeps its seed
///   index. Filter non-finite values before calling if they are possible in
///   your data.
///
/// ### Params
///
/// * `a` - The minuend slice. Must be non-empty and shorter than `2^24`.
/// * `b` - The subtrahend slice, same length as `a`.
///
/// ### Returns
///
/// `(index, value)` of the smallest `a[i] - b[i]`, ties to the lowest index.
#[inline]
pub fn argmin_diff_simd_f32(a: &[f32], b: &[f32]) -> (usize, f32) {
    debug_assert_eq!(a.len(), b.len(), "slices must match in length");
    debug_assert!(!a.is_empty(), "argmin over an empty slice is undefined");
    debug_assert!(
        a.len() < (1usize << 24),
        "lane indices are carried as f32 and stop being exact at 2^24"
    );

    #[cfg(target_arch = "x86_64")]
    // SAFETY: see `sum_squares_simd_f32`.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Avx512 => argmin_diff_avx512_f32(a, b),
            SimdLevel::Avx2 => argmin_diff_avx2_f32(a, b),
            SimdLevel::Sse => argmin_diff_sse_f32(a, b),
            SimdLevel::Scalar => argmin_diff_scalar_f32(a, b),
        }
    }

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64, so the feature is unconditionally
    // present and `detect_simd_level` always reports `Sse` here.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Sse => argmin_diff_neon_f32(a, b),
            _ => argmin_diff_scalar_f32(a, b),
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    argmin_diff_scalar_f32(a, b)
}

//////////
// Sums //
//////////

/////////
// f32 //
/////////

/// SIMD sum of a slice of f32 (scalar)
///
/// ### Params
///
/// * `a` - The slice of f32 values to sum.
///
/// ### Returns
///
/// Sum
#[inline(always)]
fn sum_scalar_f32(a: &[f32]) -> f32 {
    a.iter().sum()
}

/// SIMD sum of a slice of f32 (128-bit)
///
/// ### Params
///
/// * `a` - The slice of f32 values to sum.
///
/// ### Returns
///
/// Sum
#[inline(always)]
fn sum_sse_f32(a: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = f32x4::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        for i in 0..chunks {
            let va = f32x4::from(*(a_ptr.add(i * 4) as *const [f32; 4]));
            acc += va;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        sum += a[i];
    }
    sum
}

/// SIMD sum of a slice of f32 (256-bit)
///
/// ### Params
///
/// * `a` - The slice of f32 values to sum.
///
/// ### Returns
///
/// Sum
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sum_avx2_f32(a: &[f32]) -> f32 {
    const W: usize = 8;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let mut acc = [_mm256_setzero_ps(); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                *acc = _mm256_add_ps(*acc, _mm256_loadu_ps(a_ptr.add(base + u * W)));
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = _mm256_add_ps(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            total = _mm256_add_ps(total, _mm256_loadu_ps(a_ptr.add(i)));
            i += W;
        }

        let mut sum = hsum_avx_f32(total);
        while i < len {
            sum += a[i];
            i += 1;
        }
        sum
    }
}

/// SIMD sum of a slice of f32 (512-bit)
///
/// ### Params
///
/// * `a` - The slice of f32 values to sum.
///
/// ### Returns
///
/// Sum
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_avx512_f32(a: &[f32]) -> f32 {
    const W: usize = 16;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let mut acc = [_mm512_setzero_ps(); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                *acc = _mm512_add_ps(*acc, _mm512_loadu_ps(a_ptr.add(base + u * W)));
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = _mm512_add_ps(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            total = _mm512_add_ps(total, _mm512_loadu_ps(a_ptr.add(i)));
            i += W;
        }

        let mut sum = _mm512_reduce_add_ps(total);
        while i < len {
            sum += a[i];
            i += 1;
        }
        sum
    }
}

/// SIMD sum of a slice of f32 (dispatch)
///
/// ### Params
///
/// * `a` - The slice of f32 values to sum.
///
/// ### Returns
///
/// Sum
#[inline]
pub fn sum_simd_f32(a: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: see `sum_squares_simd_f32`.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Avx512 => sum_avx512_f32(a),
            SimdLevel::Avx2 => sum_avx2_f32(a),
            SimdLevel::Sse => sum_sse_f32(a),
            SimdLevel::Scalar => sum_scalar_f32(a),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    match detect_simd_level() {
        SimdLevel::Sse => sum_sse_f32(a),
        _ => sum_scalar_f32(a),
    }
}

/////////
// f64 //
/////////

/// SIMD sum of a slice of f64 (scalar)
///
/// ### Params
///
/// * `a` - The slice of f64 values to sum.
///
/// ### Returns
///
/// Sum
#[inline(always)]
fn sum_scalar_f64(a: &[f64]) -> f64 {
    a.iter().sum()
}

/// SIMD sum of a slice of f64 (128-bit)
///
/// ### Params
///
/// * `a` - The slice of f64 values to sum.
///
/// ### Returns
///
/// Sum
#[inline(always)]
fn sum_sse_f64(a: &[f64]) -> f64 {
    let len = a.len();
    let chunks = len / 2;
    let mut acc = f64x2::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        for i in 0..chunks {
            let va = f64x2::from(*(a_ptr.add(i * 2) as *const [f64; 2]));
            acc += va;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 2)..len {
        sum += a[i];
    }
    sum
}

/// SIMD sum of a slice of f64 (256-bit)
///
/// ### Params
///
/// * `a` - The slice of f64 values to sum.
///
/// ### Returns
///
/// Sum
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sum_avx2_f64(a: &[f64]) -> f64 {
    const W: usize = 4;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let mut acc = [_mm256_setzero_pd(); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                *acc = _mm256_add_pd(*acc, _mm256_loadu_pd(a_ptr.add(base + u * W)));
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = _mm256_add_pd(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            total = _mm256_add_pd(total, _mm256_loadu_pd(a_ptr.add(i)));
            i += W;
        }

        let mut sum = hsum_avx_f64(total);
        while i < len {
            sum += a[i];
            i += 1;
        }
        sum
    }
}

/// SIMD sum of a slice of f64 (512-bit)
///
/// ### Params
///
/// * `a` - The slice of f64 values to sum.
///
/// ### Returns
///
/// Sum
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_avx512_f64(a: &[f64]) -> f64 {
    const W: usize = 8;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let mut acc = [_mm512_setzero_pd(); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                *acc = _mm512_add_pd(*acc, _mm512_loadu_pd(a_ptr.add(base + u * W)));
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = _mm512_add_pd(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            total = _mm512_add_pd(total, _mm512_loadu_pd(a_ptr.add(i)));
            i += W;
        }

        let mut sum = _mm512_reduce_add_pd(total);
        while i < len {
            sum += a[i];
            i += 1;
        }
        sum
    }
}

/// SIMD sum of a slice of f64 (dispatch)
///
/// ### Params
///
/// * `a` - The slice of f64 values to sum.
///
/// ### Returns
///
/// Sum
#[inline]
pub fn sum_simd_f64(a: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: see `sum_squares_simd_f32`.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Avx512 => sum_avx512_f64(a),
            SimdLevel::Avx2 => sum_avx2_f64(a),
            SimdLevel::Sse => sum_sse_f64(a),
            SimdLevel::Scalar => sum_scalar_f64(a),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    match detect_simd_level() {
        SimdLevel::Sse => sum_sse_f64(a),
        _ => sum_scalar_f64(a),
    }
}

//////////////
// Variance //
//////////////

/////////
// f32 //
/////////

/// SIMD sum of squared deviations of a slice of f32 (scalar)
///
/// ### Params
///
/// * `a` - The slice of f32 values to calculate variance for.
/// * `mean` - The mean of the values in `a`.
///
/// ### Returns
///
/// Sum of squared deviations from the mean
#[inline(always)]
fn sum_squared_dev_scalar_f32(a: &[f32], mean: f32) -> f32 {
    a.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
}

/// SIMD sum of squared deviations of a slice of f32 (128-bit)
///
/// ### Params
///
/// * `a` - The slice of f32 values to calculate variance for.
/// * `mean` - The mean of the values in `a`.
///
/// ### Returns
///
/// Sum of squared deviations from the mean
#[inline(always)]
fn sum_squared_dev_sse_f32(a: &[f32], mean: f32) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = f32x4::ZERO;
    let mean_vec = f32x4::splat(mean);
    unsafe {
        let a_ptr = a.as_ptr();
        for i in 0..chunks {
            let va = f32x4::from(*(a_ptr.add(i * 4) as *const [f32; 4]));
            let diff = va - mean_vec;
            acc += diff * diff;
        }
    }
    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        let diff = a[i] - mean;
        sum += diff * diff;
    }
    sum
}

/// SIMD sum of squared deviations of a slice of f32 (256-bit)
///
/// ### Params
///
/// * `a` - The slice of f32 values to calculate variance for.
/// * `mean` - The mean of the values in `a`.
///
/// ### Returns
///
/// Sum of squared deviations from the mean
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sum_squared_dev_avx2_f32(a: &[f32], mean: f32) -> f32 {
    const W: usize = 8;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let mean_vec = _mm256_set1_ps(mean);
        let mut acc = [_mm256_setzero_ps(); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                let diff = _mm256_sub_ps(_mm256_loadu_ps(a_ptr.add(base + u * W)), mean_vec);
                *acc = _mm256_fmadd_ps(diff, diff, *acc);
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = _mm256_add_ps(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let diff = _mm256_sub_ps(_mm256_loadu_ps(a_ptr.add(i)), mean_vec);
            total = _mm256_fmadd_ps(diff, diff, total);
            i += W;
        }

        let mut sum = hsum_avx_f32(total);
        while i < len {
            let diff = a[i] - mean;
            sum += diff * diff;
            i += 1;
        }
        sum
    }
}

/// SIMD sum of squared deviations of a slice of f32 (512-bit)
///
/// ### Params
///
/// * `a` - The slice of f32 values to calculate variance for.
/// * `mean` - The mean of the values in `a`.
///
/// ### Returns
///
/// Sum of squared deviations from the mean
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_squared_dev_avx512_f32(a: &[f32], mean: f32) -> f32 {
    const W: usize = 16;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let mean_vec = _mm512_set1_ps(mean);
        let mut acc = [_mm512_setzero_ps(); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                let diff = _mm512_sub_ps(_mm512_loadu_ps(a_ptr.add(base + u * W)), mean_vec);
                *acc = _mm512_fmadd_ps(diff, diff, *acc);
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = _mm512_add_ps(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let diff = _mm512_sub_ps(_mm512_loadu_ps(a_ptr.add(i)), mean_vec);
            total = _mm512_fmadd_ps(diff, diff, total);
            i += W;
        }

        let mut sum = _mm512_reduce_add_ps(total);
        while i < len {
            let diff = a[i] - mean;
            sum += diff * diff;
            i += 1;
        }
        sum
    }
}

/// SIMD sum of squared deviations of a slice of f32 (dispatch)
///
/// ### Params
///
/// * `a` - The slice of f32 values to calculate variance for.
/// * `mean` - The mean of the values in `a`.
///
/// ### Returns
///
/// Sum of squared deviations from the mean
#[inline]
pub fn sum_squared_dev_simd_f32(a: &[f32], mean: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: see `sum_squares_simd_f32`.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Avx512 => sum_squared_dev_avx512_f32(a, mean),
            SimdLevel::Avx2 => sum_squared_dev_avx2_f32(a, mean),
            SimdLevel::Sse => sum_squared_dev_sse_f32(a, mean),
            SimdLevel::Scalar => sum_squared_dev_scalar_f32(a, mean),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    match detect_simd_level() {
        SimdLevel::Sse => sum_squared_dev_sse_f32(a, mean),
        _ => sum_squared_dev_scalar_f32(a, mean),
    }
}

/////////
// f64 //
/////////

/// SIMD sum of squared deviations of a slice of f64 (scalar)
///
/// ### Params
///
/// * `a` - The slice of f64 values to calculate variance for.
/// * `mean` - The mean of the values in `a`.
///
/// ### Returns
///
/// Sum of squared deviations from the mean
#[inline(always)]
fn sum_squared_dev_scalar_f64(a: &[f64], mean: f64) -> f64 {
    a.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
}

/// SIMD sum of squared deviations of a slice of f64 (128-bit)
///
/// ### Params
///
/// * `a` - The slice of f64 values to calculate variance for.
/// * `mean` - The mean of the values in `a`.
///
/// ### Returns
///
/// Sum of squared deviations from the mean
#[inline(always)]
fn sum_squared_dev_sse_f64(a: &[f64], mean: f64) -> f64 {
    let len = a.len();
    let chunks = len / 2;
    let mut acc = f64x2::ZERO;
    let mean_vec = f64x2::splat(mean);

    unsafe {
        let a_ptr = a.as_ptr();
        for i in 0..chunks {
            let va = f64x2::from(*(a_ptr.add(i * 2) as *const [f64; 2]));
            let diff = va - mean_vec;
            acc += diff * diff;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 2)..len {
        let diff = a[i] - mean;
        sum += diff * diff;
    }
    sum
}

/// SIMD sum of squared deviations of a slice of f64 (256-bit)
///
/// ### Params
///
/// * `a` - The slice of f64 values to calculate variance for.
/// * `mean` - The mean of the values in `a`.
///
/// ### Returns
///
/// Sum of squared deviations from the mean
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sum_squared_dev_avx2_f64(a: &[f64], mean: f64) -> f64 {
    const W: usize = 4;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let mean_vec = _mm256_set1_pd(mean);
        let mut acc = [_mm256_setzero_pd(); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                let diff = _mm256_sub_pd(_mm256_loadu_pd(a_ptr.add(base + u * W)), mean_vec);
                *acc = _mm256_fmadd_pd(diff, diff, *acc);
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = _mm256_add_pd(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let diff = _mm256_sub_pd(_mm256_loadu_pd(a_ptr.add(i)), mean_vec);
            total = _mm256_fmadd_pd(diff, diff, total);
            i += W;
        }

        let mut sum = hsum_avx_f64(total);
        while i < len {
            let diff = a[i] - mean;
            sum += diff * diff;
            i += 1;
        }
        sum
    }
}

/// SIMD sum of squared deviations of a slice of f64 (512-bit)
///
/// ### Params
///
/// * `a` - The slice of f64 values to calculate variance for.
/// * `mean` - The mean of the values in `a`.
///
/// ### Returns
///
/// Sum of squared deviations from the mean
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_squared_dev_avx512_f64(a: &[f64], mean: f64) -> f64 {
    const W: usize = 8;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let mean_vec = _mm512_set1_pd(mean);
        let mut acc = [_mm512_setzero_pd(); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                let diff = _mm512_sub_pd(_mm512_loadu_pd(a_ptr.add(base + u * W)), mean_vec);
                *acc = _mm512_fmadd_pd(diff, diff, *acc);
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = _mm512_add_pd(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let diff = _mm512_sub_pd(_mm512_loadu_pd(a_ptr.add(i)), mean_vec);
            total = _mm512_fmadd_pd(diff, diff, total);
            i += W;
        }

        let mut sum = _mm512_reduce_add_pd(total);
        while i < len {
            let diff = a[i] - mean;
            sum += diff * diff;
            i += 1;
        }
        sum
    }
}

/// SIMD sum of squared deviations of a slice of f64 (dispatch)
///
/// ### Params
///
/// * `a` - The slice of f64 values to calculate variance for.
/// * `mean` - The mean of the values in `a`.
///
/// ### Returns
///
/// Sum of squared deviations from the mean
#[inline]
pub fn sum_squared_dev_simd_f64(a: &[f64], mean: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: see `sum_squares_simd_f32`.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Avx512 => sum_squared_dev_avx512_f64(a, mean),
            SimdLevel::Avx2 => sum_squared_dev_avx2_f64(a, mean),
            SimdLevel::Sse => sum_squared_dev_sse_f64(a, mean),
            SimdLevel::Scalar => sum_squared_dev_scalar_f64(a, mean),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    match detect_simd_level() {
        SimdLevel::Sse => sum_squared_dev_sse_f64(a, mean),
        _ => sum_squared_dev_scalar_f64(a, mean),
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::prelude::*;
    use rand::rngs::StdRng;

    /// Lengths sweeping the 2-, 4-, 8- and 16-wide boundaries and the unrolled
    /// block sizes (4x4 = 16, 4x8 = 32, 4x16 = 64), so both the block loop and
    /// every tail path get exercised.
    const LENGTHS: [usize; 22] = [
        1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 257, 666, 1000,
    ];

    /// The vectorised subtract-and-argmin must agree with the scalar scan it
    /// replaced, exactly.
    ///
    /// The low-cardinality case is the important one: ties are where a
    /// lane-wise reduction is most likely to disagree, and with `UNROLL`
    /// independent accumulators a tie can now also straddle two accumulators
    /// whose minima came from different blocks.
    #[test]
    fn test_argmin_diff_simd_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(42);

        for len in LENGTHS {
            for tied in [false, true] {
                let (a, b): (Vec<f32>, Vec<f32>) = (0..len)
                    .map(|_| {
                        if tied {
                            // Few distinct differences, so ties are frequent.
                            (rng.random_range(0..3) as f32, 0.0f32)
                        } else {
                            (
                                rng.random::<f32>() * 2.0 - 1.0,
                                rng.random::<f32>() * 2.0 - 1.0,
                            )
                        }
                    })
                    .unzip();

                let (simd_idx, simd_val) = argmin_diff_simd_f32(&a, &b);
                let (scalar_idx, scalar_val) = argmin_diff_scalar_f32(&a, &b);

                assert_eq!(
                    simd_idx, scalar_idx,
                    "index mismatch at len {} (tied {:?})",
                    len, tied
                );
                assert_eq!(
                    simd_val, scalar_val,
                    "value mismatch at len {} (tied {:?})",
                    len, tied
                );
            }
        }
    }

    /// Every width must agree with the scalar reference regardless of what the
    /// runtime dispatch picks on this machine, so the arms that are not
    /// selected here still get covered wherever the hardware allows it.
    #[test]
    fn test_argmin_diff_widths_agree() {
        let mut rng = StdRng::seed_from_u64(7);
        for len in LENGTHS {
            for tied in [false, true] {
                let (a, b): (Vec<f32>, Vec<f32>) = (0..len)
                    .map(|_| {
                        if tied {
                            (rng.random_range(0..3) as f32, 0.0f32)
                        } else {
                            (rng.random::<f32>(), rng.random::<f32>())
                        }
                    })
                    .unzip();

                let reference = argmin_diff_scalar_f32(&a, &b);

                #[cfg(target_arch = "aarch64")]
                unsafe {
                    assert_eq!(
                        argmin_diff_neon_f32(&a, &b),
                        reference,
                        "neon at len {} (tied {:?})",
                        len,
                        tied
                    );
                }

                #[cfg(target_arch = "x86_64")]
                unsafe {
                    if is_x86_feature_detected!("sse4.1") {
                        assert_eq!(
                            argmin_diff_sse_f32(&a, &b),
                            reference,
                            "sse at len {} (tied {:?})",
                            len,
                            tied
                        );
                    }
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        assert_eq!(
                            argmin_diff_avx2_f32(&a, &b),
                            reference,
                            "avx2 at len {} (tied {:?})",
                            len,
                            tied
                        );
                    }
                    if is_x86_feature_detected!("avx512f") {
                        assert_eq!(
                            argmin_diff_avx512_f32(&a, &b),
                            reference,
                            "avx512 at len {} (tied {:?})",
                            len,
                            tied
                        );
                    }
                }
            }
        }
    }

    /// Every reduction arm must agree with its scalar reference on this
    /// machine. Horizontal order differs between widths, so this is a
    /// tolerance check rather than an exact one; the argmin above is the only
    /// kernel where bit-exact agreement is required.
    ///
    /// On aarch64 only the 128-bit arm runs; the x86 arms are covered wherever
    /// CI provides the hardware.
    #[test]
    fn test_reduction_widths_agree() {
        let mut rng = StdRng::seed_from_u64(99);

        for len in LENGTHS {
            let a32: Vec<f32> = (0..len).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            let b32: Vec<f32> = (0..len).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            let a64: Vec<f64> = a32.iter().map(|&x| x as f64).collect();
            let b64: Vec<f64> = b32.iter().map(|&x| x as f64).collect();
            let mean32 = sum_scalar_f32(&a32) / len as f32;
            let mean64 = sum_scalar_f64(&a64) / len as f64;

            assert_relative_eq!(
                sum_squares_sse_f32(&a32),
                sum_squares_scalar_f32(&a32),
                epsilon = 1e-4,
                max_relative = 1e-5
            );
            assert_relative_eq!(
                dot_sse_f32(&a32, &b32),
                dot_scalar_f32(&a32, &b32),
                epsilon = 1e-4,
                max_relative = 1e-5
            );
            assert_relative_eq!(
                sum_sse_f32(&a32),
                sum_scalar_f32(&a32),
                epsilon = 1e-4,
                max_relative = 1e-5
            );
            assert_relative_eq!(
                sum_squared_dev_sse_f32(&a32, mean32),
                sum_squared_dev_scalar_f32(&a32, mean32),
                epsilon = 1e-4,
                max_relative = 1e-5
            );
            assert_relative_eq!(
                dot_sse_f64(&a64, &b64),
                dot_scalar_f64(&a64, &b64),
                epsilon = 1e-12,
                max_relative = 1e-12
            );
            assert_relative_eq!(
                sum_sse_f64(&a64),
                sum_scalar_f64(&a64),
                epsilon = 1e-12,
                max_relative = 1e-12
            );
            assert_relative_eq!(
                sum_squared_dev_sse_f64(&a64, mean64),
                sum_squared_dev_scalar_f64(&a64, mean64),
                epsilon = 1e-12,
                max_relative = 1e-12
            );

            #[cfg(target_arch = "x86_64")]
            unsafe {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    assert_relative_eq!(
                        sum_squares_avx2_f32(&a32),
                        sum_squares_scalar_f32(&a32),
                        epsilon = 1e-4,
                        max_relative = 1e-5
                    );
                    assert_relative_eq!(
                        dot_avx2_f32(&a32, &b32),
                        dot_scalar_f32(&a32, &b32),
                        epsilon = 1e-4,
                        max_relative = 1e-5
                    );
                    assert_relative_eq!(
                        sum_avx2_f32(&a32),
                        sum_scalar_f32(&a32),
                        epsilon = 1e-4,
                        max_relative = 1e-5
                    );
                    assert_relative_eq!(
                        sum_squared_dev_avx2_f32(&a32, mean32),
                        sum_squared_dev_scalar_f32(&a32, mean32),
                        epsilon = 1e-4,
                        max_relative = 1e-5
                    );
                    assert_relative_eq!(
                        dot_avx2_f64(&a64, &b64),
                        dot_scalar_f64(&a64, &b64),
                        epsilon = 1e-12,
                        max_relative = 1e-12
                    );
                    assert_relative_eq!(
                        sum_avx2_f64(&a64),
                        sum_scalar_f64(&a64),
                        epsilon = 1e-12,
                        max_relative = 1e-12
                    );
                    assert_relative_eq!(
                        sum_squared_dev_avx2_f64(&a64, mean64),
                        sum_squared_dev_scalar_f64(&a64, mean64),
                        epsilon = 1e-12,
                        max_relative = 1e-12
                    );
                }

                if is_x86_feature_detected!("avx512f") {
                    assert_relative_eq!(
                        sum_squares_avx512_f32(&a32),
                        sum_squares_scalar_f32(&a32),
                        epsilon = 1e-4,
                        max_relative = 1e-5
                    );
                    assert_relative_eq!(
                        dot_avx512_f32(&a32, &b32),
                        dot_scalar_f32(&a32, &b32),
                        epsilon = 1e-4,
                        max_relative = 1e-5
                    );
                    assert_relative_eq!(
                        sum_avx512_f32(&a32),
                        sum_scalar_f32(&a32),
                        epsilon = 1e-4,
                        max_relative = 1e-5
                    );
                    assert_relative_eq!(
                        sum_squared_dev_avx512_f32(&a32, mean32),
                        sum_squared_dev_scalar_f32(&a32, mean32),
                        epsilon = 1e-4,
                        max_relative = 1e-5
                    );
                    assert_relative_eq!(
                        dot_avx512_f64(&a64, &b64),
                        dot_scalar_f64(&a64, &b64),
                        epsilon = 1e-12,
                        max_relative = 1e-12
                    );
                    assert_relative_eq!(
                        sum_avx512_f64(&a64),
                        sum_scalar_f64(&a64),
                        epsilon = 1e-12,
                        max_relative = 1e-12
                    );
                    assert_relative_eq!(
                        sum_squared_dev_avx512_f64(&a64, mean64),
                        sum_squared_dev_scalar_f64(&a64, mean64),
                        epsilon = 1e-12,
                        max_relative = 1e-12
                    );
                }
            }
        }
    }
}
