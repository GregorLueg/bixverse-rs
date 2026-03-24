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

///////////
// Sums  //
///////////

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

//////////////////
// Sum squares  //
//////////////////

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

//////////////
// Variance //
//////////////

/// SIMD variance of a slice of f32 (scalar)
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
fn variance_scalar_f32(a: &[f32], mean: f32) -> f32 {
    a.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
}

/// SIMD variance of a slice of f32 (128-bit)
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
fn variance_sse_f32(a: &[f32], mean: f32) -> f32 {
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

/// SIMD variance of a slice of f32 (256-bit)
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
fn variance_avx2_f32(a: &[f32], mean: f32) -> f32 {
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

/// SIMD variance of a slice of f32 (512-bit)
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
fn variance_avx512_f32(a: &[f32], mean: f32) -> f32 {
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

/// SIMD variance of a slice of f32 (512-bit fallback)
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
fn variance_avx512_f32(a: &[f32], mean: f32) -> f32 {
    variance_avx2_f32(a, mean)
}

/// SIMD variance of a slice of f32 (dispatch)
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
pub fn variance_simd_f32(a: &[f32], mean: f32) -> f32 {
    match detect_simd_level() {
        SimdLevel::Avx512 => variance_avx512_f32(a, mean),
        SimdLevel::Avx2 => variance_avx2_f32(a, mean),
        SimdLevel::Sse => variance_sse_f32(a, mean),
        SimdLevel::Scalar => variance_scalar_f32(a, mean),
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

impl BixverseSimd for f32 {
    #[inline]
    fn bxv_dot_simd(a: &[f32], b: &[f32]) -> f32 {
        dot_simd_f32(a, b)
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

impl BixverseSimd for f64 {
    #[inline]
    fn bxv_dot_simd(a: &[f64], b: &[f64]) -> f64 {
        dot_simd_f64(a, b)
    }
}
