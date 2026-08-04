//! Targeted SIMD implementations to accelerate specific hot loops in bixverse

use std::sync::OnceLock;
use wide::{f32x4, f32x8, f64x2, f64x4};

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
pub fn detect_simd_level() -> SimdLevel {
    *SIMD_LEVEL.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return SimdLevel::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
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

//////////////////////////////////
// f32-specific implementations //
//////////////////////////////////

//////////
// Sums //
//////////

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
#[inline(always)]
fn sum_squares_avx2_f32(a: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let mut acc = f32x8::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        for i in 0..chunks {
            let va = f32x8::from(*(a_ptr.add(i * 8) as *const [f32; 8]));
            acc += va * va;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 8)..len {
        sum += a[i] * a[i];
    }
    sum
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
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn sum_squares_avx512_f32(a: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 16;

    unsafe {
        let mut acc = _mm512_setzero_ps();

        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            acc = _mm512_fmadd_ps(va, va, acc);
        }

        let mut sum = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..len {
            sum += a[i] * a[i];
        }
        sum
    }
}

/// SIMD squared sum of a slice of f32 (512-bit fallback)
///
/// ### Params
///
/// * `a` - The slice of f32 values to sum.
///
/// ### Returns
///
/// Squared sum
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn sum_squares_avx512_f32(a: &[f32]) -> f32 {
    sum_squares_avx2_f32(a)
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
    match detect_simd_level() {
        SimdLevel::Avx512 => sum_squares_avx512_f32(a),
        SimdLevel::Avx2 => sum_squares_avx2_f32(a),
        SimdLevel::Sse => sum_squares_sse_f32(a),
        SimdLevel::Scalar => sum_squares_scalar_f32(a),
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
#[inline(always)]
fn dot_avx2_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let mut acc = f32x8::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        for i in 0..chunks {
            let va = f32x8::from(*(a_ptr.add(i * 8) as *const [f32; 8]));
            let vb = f32x8::from(*(b_ptr.add(i * 8) as *const [f32; 8]));
            acc += va * vb;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 8)..len {
        sum += a[i] * b[i];
    }
    sum
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
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn dot_avx512_f32(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 16;

    unsafe {
        let mut acc = _mm512_setzero_ps();
        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            acc = _mm512_fmadd_ps(va, vb, acc);
        }

        let mut sum = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..len {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// SIMD dot product of two slices of f32 (512-bit fallback)
///
/// ### Params
///
/// * `a` - The first slice of f32 values.
/// * `b` - The second slice of f32 values.
///
/// ### Returns
///
/// Dot product
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn dot_avx512_f32(a: &[f32], b: &[f32]) -> f32 {
    dot_avx2_f32(a, b)
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

/// Fused subtract-and-argmin over two slices of f32 (128-bit)
///
/// ### Params
///
/// * `a` - The minuend slice.
/// * `b` - The subtrahend slice, same length as `a`.
///
/// ### Returns
///
/// `(index, value)` of the smallest `a[i] - b[i]`, ties to the lowest index.
#[inline]
fn argmin_diff_sse_f32(a: &[f32], b: &[f32]) -> (usize, f32) {
    let len = a.len();
    let chunks = len / 4;

    let mut best = f32x4::from(f32::INFINITY);
    let mut best_idx = f32x4::ZERO;
    let mut lane = f32x4::from([0.0, 1.0, 2.0, 3.0]);
    let step = f32x4::from(4.0);

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        for i in 0..chunks {
            let va = f32x4::from(*(a_ptr.add(i * 4) as *const [f32; 4]));
            let vb = f32x4::from(*(b_ptr.add(i * 4) as *const [f32; 4]));
            let diff = va - vb;
            // Strict `<` keeps the earliest lane on a tie, matching the scalar
            // scan the callers replaced.
            let mask = diff.simd_lt(best);
            best = mask.blend(diff, best);
            best_idx = mask.blend(lane, best_idx);
            lane += step;
        }
    }

    let (mut min_idx, mut min_val) = reduce_lane_argmin_f32(&best.to_array(), &best_idx.to_array());

    for i in (chunks * 4)..len {
        let diff = a[i] - b[i];
        if diff < min_val {
            min_val = diff;
            min_idx = i;
        }
    }

    (min_idx, min_val)
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
#[inline]
fn argmin_diff_avx2_f32(a: &[f32], b: &[f32]) -> (usize, f32) {
    let len = a.len();
    let chunks = len / 8;

    let mut best = f32x8::from(f32::INFINITY);
    let mut best_idx = f32x8::ZERO;
    let mut lane = f32x8::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let step = f32x8::from(8.0);

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        for i in 0..chunks {
            let va = f32x8::from(*(a_ptr.add(i * 8) as *const [f32; 8]));
            let vb = f32x8::from(*(b_ptr.add(i * 8) as *const [f32; 8]));
            let diff = va - vb;
            let mask = diff.simd_lt(best);
            best = mask.blend(diff, best);
            best_idx = mask.blend(lane, best_idx);
            lane += step;
        }
    }

    let (mut min_idx, mut min_val) = reduce_lane_argmin_f32(&best.to_array(), &best_idx.to_array());

    for i in (chunks * 8)..len {
        let diff = a[i] - b[i];
        if diff < min_val {
            min_val = diff;
            min_idx = i;
        }
    }

    (min_idx, min_val)
}

/// Fused subtract-and-argmin over two slices of f32 (dispatch)
///
/// Computes `argmin_i (a[i] - b[i])` without materialising the difference. The
/// scalar form of this loop does not auto-vectorise, because the running
/// minimum is a data-dependent branch, so it is the one pass in the Frank-Wolfe
/// column update that stays scalar unless it is written out by hand.
///
/// AVX-512 dispatches to the 256-bit path, matching the rest of this module.
///
/// Two caveats, neither reachable from the Frank-Wolfe callers but both visible
/// to anyone else calling this:
///
/// - Lane indices are carried as `f32`, exact only below `2^24`. Longer slices
///   would return a rounded index.
/// - `NaN` is not handled consistently across the arms. The scalar path seeds
///   from `a[0] - b[0]`, so a `NaN` there poisons every later comparison and it
///   returns index 0; the vector paths seed at `+INFINITY` and never blend a
///   `NaN` in, so they return the minimum over the finite differences. Filter
///   `NaN` before calling if it is possible in your data.
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

    match detect_simd_level() {
        SimdLevel::Avx512 | SimdLevel::Avx2 => argmin_diff_avx2_f32(a, b),
        SimdLevel::Sse => argmin_diff_sse_f32(a, b),
        SimdLevel::Scalar => argmin_diff_scalar_f32(a, b),
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
    match detect_simd_level() {
        SimdLevel::Avx512 => dot_avx512_f32(a, b),
        SimdLevel::Avx2 => dot_avx2_f32(a, b),
        SimdLevel::Sse => dot_sse_f32(a, b),
        SimdLevel::Scalar => dot_scalar_f32(a, b),
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
#[inline(always)]
fn dot_avx2_f64(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = f64x4::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        for i in 0..chunks {
            let va = f64x4::from(*(a_ptr.add(i * 4) as *const [f64; 4]));
            let vb = f64x4::from(*(b_ptr.add(i * 4) as *const [f64; 4]));
            acc += va * vb;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        sum += a[i] * b[i];
    }
    sum
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
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn dot_avx512_f64(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;

    unsafe {
        let mut acc = _mm512_setzero_pd();
        for i in 0..chunks {
            let va = _mm512_loadu_pd(a.as_ptr().add(i * 8));
            let vb = _mm512_loadu_pd(b.as_ptr().add(i * 8));
            acc = _mm512_fmadd_pd(va, vb, acc);
        }

        let mut sum = _mm512_reduce_add_pd(acc);
        for i in (chunks * 8)..len {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// SIMD dot product of two slices of f64 (512-bit fallback)
///
/// ### Params
///
/// * `a` - The first slice of f64 values.
/// * `b` - The second slice of f64 values.
///
/// ### Returns
///
/// Dot product
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn dot_avx512_f64(a: &[f64], b: &[f64]) -> f64 {
    dot_avx2_f64(a, b)
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
    match detect_simd_level() {
        SimdLevel::Avx512 => dot_avx512_f64(a, b),
        SimdLevel::Avx2 => dot_avx2_f64(a, b),
        SimdLevel::Sse => dot_sse_f64(a, b),
        SimdLevel::Scalar => dot_scalar_f64(a, b),
    }
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
#[inline(always)]
fn sum_avx2_f32(a: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let mut acc = f32x8::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        for i in 0..chunks {
            let va = f32x8::from(*(a_ptr.add(i * 8) as *const [f32; 8]));
            acc += va;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 8)..len {
        sum += a[i];
    }
    sum
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
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn sum_avx512_f32(a: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 16;

    unsafe {
        let mut acc = _mm512_setzero_ps();

        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            acc = _mm512_add_ps(acc, va);
        }

        let mut sum = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..len {
            sum += a[i];
        }
        sum
    }
}

/// SIMD sum of a slice of f32 (512-bit fallback)
///
/// ### Params
///
/// * `a` - The slice of f32 values to sum.
///
/// ### Returns
///
/// Sum
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn sum_avx512_f32(a: &[f32]) -> f32 {
    sum_avx2_f32(a)
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
    match detect_simd_level() {
        SimdLevel::Avx512 => sum_avx512_f32(a),
        SimdLevel::Avx2 => sum_avx2_f32(a),
        SimdLevel::Sse => sum_sse_f32(a),
        SimdLevel::Scalar => sum_scalar_f32(a),
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
#[inline(always)]
fn sum_avx2_f64(a: &[f64]) -> f64 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = f64x4::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        for i in 0..chunks {
            let va = f64x4::from(*(a_ptr.add(i * 4) as *const [f64; 4]));
            acc += va;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        sum += a[i];
    }
    sum
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
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn sum_avx512_f64(a: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;

    unsafe {
        let mut acc = _mm512_setzero_pd();

        for i in 0..chunks {
            let va = _mm512_loadu_pd(a.as_ptr().add(i * 8));
            acc = _mm512_add_pd(acc, va);
        }

        let mut sum = _mm512_reduce_add_pd(acc);
        for i in (chunks * 8)..len {
            sum += a[i];
        }
        sum
    }
}

/// SIMD sum of a slice of f64 (512-bit fallback)
///
/// ### Params
///
/// * `a` - The slice of f64 values to sum.
///
/// ### Returns
///
/// Sum
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn sum_avx512_f64(a: &[f64]) -> f64 {
    sum_avx2_f64(a)
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
    match detect_simd_level() {
        SimdLevel::Avx512 => sum_avx512_f64(a),
        SimdLevel::Avx2 => sum_avx2_f64(a),
        SimdLevel::Sse => sum_sse_f64(a),
        SimdLevel::Scalar => sum_scalar_f64(a),
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
/// Variance
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
/// Variance
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
/// Variance
#[inline(always)]
fn sum_squared_dev_avx2_f32(a: &[f32], mean: f32) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let mut acc = f32x8::ZERO;
    let mean_vec = f32x8::splat(mean);
    unsafe {
        let a_ptr = a.as_ptr();
        for i in 0..chunks {
            let va = f32x8::from(*(a_ptr.add(i * 8) as *const [f32; 8]));
            let diff = va - mean_vec;
            acc += diff * diff;
        }
    }
    let mut sum = acc.reduce_add();
    for i in (chunks * 8)..len {
        let diff = a[i] - mean;
        sum += diff * diff;
    }
    sum
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
/// Variance
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn sum_squared_dev_avx512_f32(a: &[f32], mean: f32) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let chunks = len / 16;
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let mean_vec = _mm512_set1_ps(mean);
        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let diff = _mm512_sub_ps(va, mean_vec);
            acc = _mm512_fmadd_ps(diff, diff, acc);
        }
        let mut sum = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..len {
            let diff = a[i] - mean;
            sum += diff * diff;
        }
        sum
    }
}

/// SIMD sum of squared deviations of a slice of f32 (512-bit fallback)
///
/// ### Params
///
/// * `a` - The slice of f32 values to calculate variance for.
/// * `mean` - The mean of the values in `a`.
///
/// ### Returns
///
/// Variance
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn sum_squared_dev_avx512_f32(a: &[f32], mean: f32) -> f32 {
    sum_squared_dev_avx2_f32(a, mean)
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
/// Variance
#[inline]
pub fn sum_squared_dev_simd_f32(a: &[f32], mean: f32) -> f32 {
    match detect_simd_level() {
        SimdLevel::Avx512 => sum_squared_dev_avx512_f32(a, mean),
        SimdLevel::Avx2 => sum_squared_dev_avx2_f32(a, mean),
        SimdLevel::Sse => sum_squared_dev_sse_f32(a, mean),
        SimdLevel::Scalar => sum_squared_dev_scalar_f32(a, mean),
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
#[inline(always)]
fn sum_squared_dev_avx2_f64(a: &[f64], mean: f64) -> f64 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = f64x4::ZERO;
    let mean_vec = f64x4::splat(mean);

    unsafe {
        let a_ptr = a.as_ptr();
        for i in 0..chunks {
            let va = f64x4::from(*(a_ptr.add(i * 4) as *const [f64; 4]));
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
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn sum_squared_dev_avx512_f64(a: &[f64], mean: f64) -> f64 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;

    unsafe {
        let mut acc = _mm512_setzero_pd();
        let mean_vec = _mm512_set1_pd(mean);

        for i in 0..chunks {
            let va = _mm512_loadu_pd(a.as_ptr().add(i * 8));
            let diff = _mm512_sub_pd(va, mean_vec);
            acc = _mm512_fmadd_pd(diff, diff, acc);
        }

        let mut sum = _mm512_reduce_add_pd(acc);
        for i in (chunks * 8)..len {
            let diff = a[i] - mean;
            sum += diff * diff;
        }
        sum
    }
}

/// SIMD sum of squared deviations of a slice of f64 (512-bit fallback)
///
/// ### Params
///
/// * `a` - The slice of f64 values to calculate variance for.
/// * `mean` - The mean of the values in `a`.
///
/// ### Returns
///
/// Sum of squared deviations from the mean
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn sum_squared_dev_avx512_f64(a: &[f64], mean: f64) -> f64 {
    sum_squared_dev_avx2_f64(a, mean)
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
    match detect_simd_level() {
        SimdLevel::Avx512 => sum_squared_dev_avx512_f64(a, mean),
        SimdLevel::Avx2 => sum_squared_dev_avx2_f64(a, mean),
        SimdLevel::Sse => sum_squared_dev_sse_f64(a, mean),
        SimdLevel::Scalar => sum_squared_dev_scalar_f64(a, mean),
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand::rngs::StdRng;

    /// The vectorised subtract-and-argmin must agree with the scalar scan it
    /// replaced, exactly.
    ///
    /// Lengths are swept across the 4- and 8-wide boundaries so the tail
    /// handling is covered, and a low-cardinality case is included because ties
    /// are where a lane-wise reduction is most likely to disagree: the scalar
    /// scan uses a strict `<` and therefore keeps the lowest index.
    #[test]
    fn test_argmin_diff_simd_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(42);

        for len in [
            1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 64, 100, 257, 666,
        ] {
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

    /// Both explicit widths must agree with the scalar reference regardless of
    /// what the runtime dispatch picks on this machine, so the arm that is not
    /// selected here still gets covered.
    #[test]
    fn test_argmin_diff_widths_agree() {
        let mut rng = StdRng::seed_from_u64(7);
        for len in [1usize, 6, 13, 32, 129, 666] {
            let a: Vec<f32> = (0..len).map(|_| rng.random::<f32>()).collect();
            let b: Vec<f32> = (0..len).map(|_| rng.random::<f32>()).collect();

            let reference = argmin_diff_scalar_f32(&a, &b);
            assert_eq!(argmin_diff_sse_f32(&a, &b), reference, "sse at len {}", len);
            assert_eq!(
                argmin_diff_avx2_f32(&a, &b),
                reference,
                "avx2 at len {}",
                len
            );
        }
    }
}
