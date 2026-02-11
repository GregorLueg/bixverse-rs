use std::sync::OnceLock;
use wide::{f32x4, f32x8};

// Enum for the different architectures and potential SIMD levels
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

/////////////////////////////
// General implementations //
/////////////////////////////

/////////
// f32 //
/////////

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
