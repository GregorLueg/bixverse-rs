//! SIMD specifically designed for single cell applications
//!
//! Same split as `crate::utils::simd`: 128-bit arms go through `wide`, which
//! is real on both x86_64 (SSE2) and aarch64 (NEON), while the 256- and
//! 512-bit arms are hand-written intrinsics behind `#[target_feature]` so they
//! survive a stock build. See that module's header for why `wide`'s wider
//! types cannot stand in.

use wide::f32x4;

use crate::utils::simd::{SimdLevel, detect_simd_level};

// Only the 256- and 512-bit reduction arms carry accumulators, and those are
// x86-only.
#[cfg(target_arch = "x86_64")]
use crate::utils::simd::UNROLL;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/////////////
// Hotspot //
/////////////

///////////////////////////////
// Fused multiply-square-sum //
///////////////////////////////

/// SIMD-fused multiply-square-sum (scalar)
///
/// ### Params
///
/// * `a`: The first vector.
/// * `b`: The second vector.
///
/// ### Returns
///
/// The product
#[inline(always)]
fn fused_mul_square_sum_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi * bi).sum()
}

/// SIMD-fused multiply-square-sum (128-bit optimised)
///
/// ### Params
///
/// * `a`: The first vector.
/// * `b`: The second vector.
///
/// ### Returns
///
/// The product
#[inline(always)]
fn fused_mul_square_sum_sse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = f32x4::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 4;
            let va = f32x4::from(*(a_ptr.add(offset) as *const [f32; 4]));
            let vb = f32x4::from(*(b_ptr.add(offset) as *const [f32; 4]));
            acc += va * vb * vb;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        sum += a[i] * b[i] * b[i];
    }
    sum
}

/// SIMD-fused multiply-square-sum (256-bit optimised)
///
/// ### Params
///
/// * `a`: The first vector.
/// * `b`: The second vector.
///
/// ### Returns
///
/// The product
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn fused_mul_square_sum_avx2(a: &[f32], b: &[f32]) -> f32 {
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
                *acc = _mm256_fmadd_ps(_mm256_mul_ps(va, vb), vb, *acc);
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
            total = _mm256_fmadd_ps(_mm256_mul_ps(va, vb), vb, total);
            i += W;
        }

        let mut tmp = [0.0f32; W];
        _mm256_storeu_ps(tmp.as_mut_ptr(), total);
        let mut sum: f32 = tmp.iter().sum();
        while i < len {
            sum += a[i] * b[i] * b[i];
            i += 1;
        }
        sum
    }
}

/// SIMD-fused multiply-square-sum (512-bit optimised)
///
/// ### Params
///
/// * `a`: The first vector.
/// * `b`: The second vector.
///
/// ### Returns
///
/// The product
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fused_mul_square_sum_avx512(a: &[f32], b: &[f32]) -> f32 {
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
                *acc = _mm512_fmadd_ps(_mm512_mul_ps(va, vb), vb, *acc);
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
            total = _mm512_fmadd_ps(_mm512_mul_ps(va, vb), vb, total);
            i += W;
        }

        let mut sum = _mm512_reduce_add_ps(total);
        while i < len {
            sum += a[i] * b[i] * b[i];
            i += 1;
        }
        sum
    }
}

/// SIMD-fused multiply-square-sum - Dispatch
///
/// Used in compute_local_cov_max: `sum(a[i] * b[i] * b[i])`
///
/// ### Params
///
/// * `a`: The first vector.
/// * `b`: The second vector.
///
/// ### Returns
///
/// The product
#[inline]
pub fn fused_mul_square_sum_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: each arm is entered only once `detect_simd_level` has confirmed
    // the CPU reports the features that arm is compiled for.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Avx512 => fused_mul_square_sum_avx512(a, b),
            SimdLevel::Avx2 => fused_mul_square_sum_avx2(a, b),
            SimdLevel::Sse => fused_mul_square_sum_sse(a, b),
            SimdLevel::Scalar => fused_mul_square_sum_scalar(a, b),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    match detect_simd_level() {
        SimdLevel::Sse => fused_mul_square_sum_sse(a, b),
        _ => fused_mul_square_sum_scalar(a, b),
    }
}

///////////////////
// Center values //
///////////////////

/// SIMD center the values given mu and var (scalar)
///
/// ### Params
///
/// * `vals`: The values to center.
/// * `mu`: The mean values.
/// * `var`: The variance values.
#[inline(always)]
fn center_values_scalar(vals: &mut [f32], mu: &[f32], var: &[f32]) {
    for i in 0..vals.len() {
        vals[i] = (vals[i] - mu[i]) / var[i].sqrt();
    }
}

/// SIMD center the values given mu and var (128-bit)
///
/// ### Params
///
/// * `vals`: The values to center.
/// * `mu`: The mean values.
/// * `var`: The variance values.
#[inline(always)]
fn center_values_sse(vals: &mut [f32], mu: &[f32], var: &[f32]) {
    let len = vals.len();
    let chunks = len / 4;

    unsafe {
        let vals_ptr: *mut f32 = vals.as_mut_ptr();
        let mu_ptr: *const f32 = mu.as_ptr();
        let var_ptr: *const f32 = var.as_ptr();

        for i in 0..chunks {
            let offset = i * 4;
            let v = f32x4::from(*(vals_ptr.add(offset) as *const [f32; 4]));
            let m = f32x4::from(*(mu_ptr.add(offset) as *const [f32; 4]));
            let va = f32x4::from(*(var_ptr.add(offset) as *const [f32; 4]));

            let result = (v - m) / va.sqrt();
            *(vals_ptr.add(offset) as *mut [f32; 4]) = result.into();
        }
    }

    for i in (chunks * 4)..len {
        vals[i] = (vals[i] - mu[i]) / var[i].sqrt();
    }
}

/// SIMD center the values given mu and var (256-bit)
///
/// ### Params
///
/// * `vals`: The values to center.
/// * `mu`: The mean values.
/// * `var`: The variance values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn center_values_avx2(vals: &mut [f32], mu: &[f32], var: &[f32]) {
    const W: usize = 8;

    unsafe {
        let len = vals.len();
        let chunks = len / W;
        let vals_ptr = vals.as_mut_ptr();
        let mu_ptr = mu.as_ptr();
        let var_ptr = var.as_ptr();

        for i in 0..chunks {
            let off = i * W;
            let v = _mm256_loadu_ps(vals_ptr.add(off));
            let m = _mm256_loadu_ps(mu_ptr.add(off));
            let va = _mm256_loadu_ps(var_ptr.add(off));
            let result = _mm256_div_ps(_mm256_sub_ps(v, m), _mm256_sqrt_ps(va));
            _mm256_storeu_ps(vals_ptr.add(off), result);
        }

        for i in (chunks * W)..len {
            vals[i] = (vals[i] - mu[i]) / var[i].sqrt();
        }
    }
}

/// SIMD center the values given mu and var (512-bit)
///
/// ### Params
///
/// * `vals`: The values to center.
/// * `mu`: The mean values.
/// * `var`: The variance values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn center_values_avx512(vals: &mut [f32], mu: &[f32], var: &[f32]) {
    const W: usize = 16;

    unsafe {
        let len = vals.len();
        let chunks = len / W;
        let vals_ptr = vals.as_mut_ptr();
        let mu_ptr = mu.as_ptr();
        let var_ptr = var.as_ptr();

        for i in 0..chunks {
            let off = i * W;
            let v = _mm512_loadu_ps(vals_ptr.add(off));
            let m = _mm512_loadu_ps(mu_ptr.add(off));
            let va = _mm512_loadu_ps(var_ptr.add(off));
            let result = _mm512_div_ps(_mm512_sub_ps(v, m), _mm512_sqrt_ps(va));
            _mm512_storeu_ps(vals_ptr.add(off), result);
        }

        for i in (chunks * W)..len {
            vals[i] = (vals[i] - mu[i]) / var[i].sqrt();
        }
    }
}

/// SIMD center the values given mu and var (dispatch)
///
/// ### Params
///
/// * `vals`: The values to center.
/// * `mu`: The mean values.
/// * `var`: The variance values.
#[inline]
pub fn center_values_simd(vals: &mut [f32], mu: &[f32], var: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: see `fused_mul_square_sum_simd`.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Avx512 => center_values_avx512(vals, mu, var),
            SimdLevel::Avx2 => center_values_avx2(vals, mu, var),
            SimdLevel::Sse => center_values_sse(vals, mu, var),
            SimdLevel::Scalar => center_values_scalar(vals, mu, var),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    match detect_simd_level() {
        SimdLevel::Sse => center_values_sse(vals, mu, var),
        _ => center_values_scalar(vals, mu, var),
    }
}

/////////////////////////////////////
// Element-wise operations (a * b) //
/////////////////////////////////////

/// SIMD element-wise multiplication (scalar)
///
/// ### Params
///
/// * `a`: The first input array.
/// * `b`: The second input array.
/// * `out`: The output array for results.
#[inline(always)]
fn elementwise_mul_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

/// SIMD element-wise multiplication (128-bit)
///
/// ### Params
///
/// * `a`: The first input array.
/// * `b`: The second input array.
/// * `out`: The output array for results.
#[inline(always)]
fn elementwise_mul_sse(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len();
    let chunks = len / 4;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 4;
            let va = f32x4::from(*(a_ptr.add(offset) as *const [f32; 4]));
            let vb = f32x4::from(*(b_ptr.add(offset) as *const [f32; 4]));
            let result = va * vb;
            *(out_ptr.add(offset) as *mut [f32; 4]) = result.into();
        }
    }

    for i in (chunks * 4)..len {
        out[i] = a[i] * b[i];
    }
}

/// SIMD element-wise multiplication (256-bit)
///
/// ### Params
///
/// * `a`: The first input array.
/// * `b`: The second input array.
/// * `out`: The output array for results.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn elementwise_mul_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    const W: usize = 8;

    unsafe {
        let len = a.len();
        let chunks = len / W;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let off = i * W;
            let va = _mm256_loadu_ps(a_ptr.add(off));
            let vb = _mm256_loadu_ps(b_ptr.add(off));
            _mm256_storeu_ps(out_ptr.add(off), _mm256_mul_ps(va, vb));
        }

        for i in (chunks * W)..len {
            out[i] = a[i] * b[i];
        }
    }
}

/// SIMD element-wise multiplication (512-bit)
///
/// ### Params
///
/// * `a`: The first input array.
/// * `b`: The second input array.
/// * `out`: The output array for results.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn elementwise_mul_avx512(a: &[f32], b: &[f32], out: &mut [f32]) {
    const W: usize = 16;

    unsafe {
        let len = a.len();
        let chunks = len / W;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let off = i * W;
            let va = _mm512_loadu_ps(a_ptr.add(off));
            let vb = _mm512_loadu_ps(b_ptr.add(off));
            _mm512_storeu_ps(out_ptr.add(off), _mm512_mul_ps(va, vb));
        }

        for i in (chunks * W)..len {
            out[i] = a[i] * b[i];
        }
    }
}

/// SIMD element-wise multiplication (dispatch)
///
/// ### Params
///
/// * `a`: The first input array.
/// * `b`: The second input array.
/// * `out`: The output array for results.
#[inline]
pub fn elementwise_mul_simd(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: see `fused_mul_square_sum_simd`.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Avx512 => elementwise_mul_avx512(a, b, out),
            SimdLevel::Avx2 => elementwise_mul_avx2(a, b, out),
            SimdLevel::Sse => elementwise_mul_sse(a, b, out),
            SimdLevel::Scalar => elementwise_mul_scalar(a, b, out),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    match detect_simd_level() {
        SimdLevel::Sse => elementwise_mul_sse(a, b, out),
        _ => elementwise_mul_scalar(a, b, out),
    }
}

///////////////////////////////////
// Fused multiply-add: a * b + c //
///////////////////////////////////

/// SIMD fused multiply-add (scalar)
///
/// ### Params
///
/// * `a`: The first input array.
/// * `b`: The second input array.
/// * `c`: The third input array to add.
/// * `out`: The output array for results.
#[inline(always)]
fn fused_mul_add_scalar(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] * b[i] + c[i];
    }
}

/// SIMD fused multiply-add (128-bit)
///
/// ### Params
///
/// * `a`: The first input array.
/// * `b`: The second input array.
/// * `c`: The third input array to add.
/// * `out`: The output array for results.
#[inline(always)]
fn fused_mul_add_sse(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    let len = a.len();
    let chunks = len / 4;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 4;
            let va = f32x4::from(*(a_ptr.add(offset) as *const [f32; 4]));
            let vb = f32x4::from(*(b_ptr.add(offset) as *const [f32; 4]));
            let vc = f32x4::from(*(c_ptr.add(offset) as *const [f32; 4]));
            let result = va * vb + vc;
            *(out_ptr.add(offset) as *mut [f32; 4]) = result.into();
        }
    }

    for i in (chunks * 4)..len {
        out[i] = a[i] * b[i] + c[i];
    }
}

/// SIMD fused multiply-add (256-bit)
///
/// ### Params
///
/// * `a`: The first input array.
/// * `b`: The second input array.
/// * `c`: The third input array to add.
/// * `out`: The output array for results.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn fused_mul_add_avx2(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    const W: usize = 8;

    unsafe {
        let len = a.len();
        let chunks = len / W;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let off = i * W;
            let va = _mm256_loadu_ps(a_ptr.add(off));
            let vb = _mm256_loadu_ps(b_ptr.add(off));
            let vc = _mm256_loadu_ps(c_ptr.add(off));
            _mm256_storeu_ps(out_ptr.add(off), _mm256_fmadd_ps(va, vb, vc));
        }

        for i in (chunks * W)..len {
            out[i] = a[i] * b[i] + c[i];
        }
    }
}

/// SIMD fused multiply-add (512-bit)
///
/// ### Params
///
/// * `a`: The first input array.
/// * `b`: The second input array.
/// * `c`: The third input array to add.
/// * `out`: The output array for results.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fused_mul_add_avx512(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    const W: usize = 16;

    unsafe {
        let len = a.len();
        let chunks = len / W;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let off = i * W;
            let va = _mm512_loadu_ps(a_ptr.add(off));
            let vb = _mm512_loadu_ps(b_ptr.add(off));
            let vc = _mm512_loadu_ps(c_ptr.add(off));
            _mm512_storeu_ps(out_ptr.add(off), _mm512_fmadd_ps(va, vb, vc));
        }

        for i in (chunks * W)..len {
            out[i] = a[i] * b[i] + c[i];
        }
    }
}

/// SIMD fused multiply-add (dispatch)
///
/// Only the 256- and 512-bit arms genuinely fuse: they use `fmadd`, which
/// rounds once. The scalar and 128-bit arms compute `a * b` and add `c` as two
/// operations, so they round twice and can differ from the wider arms in the
/// last bit or so. Nothing here depends on that, but do not assume the arms
/// are bit-identical.
///
/// ### Params
///
/// * `a`: The first input array.
/// * `b`: The second input array.
/// * `c`: The third input array to add.
/// * `out`: The output array for results.
#[inline]
pub fn fused_mul_add_simd(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: see `fused_mul_square_sum_simd`.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Avx512 => fused_mul_add_avx512(a, b, c, out),
            SimdLevel::Avx2 => fused_mul_add_avx2(a, b, c, out),
            SimdLevel::Sse => fused_mul_add_sse(a, b, c, out),
            SimdLevel::Scalar => fused_mul_add_scalar(a, b, c, out),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    match detect_simd_level() {
        SimdLevel::Sse => fused_mul_add_sse(a, b, c, out),
        _ => fused_mul_add_scalar(a, b, c, out),
    }
}

////////////
// SCENIC //
////////////

//////////////////
// Accumulation //
//////////////////

/// Element-wise f32 accumulation (scalar fallback)
///
/// ### Params
///
/// * `dst` - Destination slice (mutated in place).
/// * `src` - Source slice to add from.
/// * `n` - Number of elements to process.
#[inline(always)]
fn accumulate_f32_scalar(dst: &mut [f32], src: &[f32], n: usize) {
    for k in 0..n {
        dst[k] += src[k];
    }
}

/// Element-wise f32 accumulation (128-bit: SSE2 / NEON)
///
/// ### Params
///
/// * `dst` - Destination slice (mutated in place).
/// * `src` - Source slice to add from.
/// * `n` - Number of elements to process.
#[inline(always)]
fn accumulate_f32_sse(dst: &mut [f32], src: &[f32], n: usize) {
    let chunks = n / 4;
    unsafe {
        let dst_ptr = dst.as_mut_ptr();
        let src_ptr = src.as_ptr();
        for i in 0..chunks {
            let off = i * 4;
            let vd = f32x4::from(*(dst_ptr.add(off) as *const [f32; 4]));
            let vs = f32x4::from(*(src_ptr.add(off) as *const [f32; 4]));
            *(dst_ptr.add(off) as *mut [f32; 4]) = (vd + vs).into();
        }
    }
    for k in (chunks * 4)..n {
        dst[k] += src[k];
    }
}

/// Element-wise f32 accumulation (256-bit: AVX2)
///
/// ### Params
///
/// * `dst` - Destination slice (mutated in place).
/// * `src` - Source slice to add from.
/// * `n` - Number of elements to process.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn accumulate_f32_avx2(dst: &mut [f32], src: &[f32], n: usize) {
    const W: usize = 8;

    unsafe {
        let chunks = n / W;
        let dst_ptr = dst.as_mut_ptr();
        let src_ptr = src.as_ptr();

        for i in 0..chunks {
            let off = i * W;
            let vd = _mm256_loadu_ps(dst_ptr.add(off));
            let vs = _mm256_loadu_ps(src_ptr.add(off));
            _mm256_storeu_ps(dst_ptr.add(off), _mm256_add_ps(vd, vs));
        }

        for k in (chunks * W)..n {
            dst[k] += src[k];
        }
    }
}

/// Element-wise f32 accumulation (512-bit: AVX-512F)
///
/// ### Params
///
/// * `dst` - Destination slice (mutated in place).
/// * `src` - Source slice to add from.
/// * `n` - Number of elements to process.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn accumulate_f32_avx512(dst: &mut [f32], src: &[f32], n: usize) {
    const W: usize = 16;

    unsafe {
        let chunks = n / W;
        let dst_ptr = dst.as_mut_ptr();
        let src_ptr = src.as_ptr();

        for i in 0..chunks {
            let off = i * W;
            let vd = _mm512_loadu_ps(dst_ptr.add(off));
            let vs = _mm512_loadu_ps(src_ptr.add(off));
            _mm512_storeu_ps(dst_ptr.add(off), _mm512_add_ps(vd, vs));
        }

        for k in (chunks * W)..n {
            dst[k] += src[k];
        }
    }
}

/// Element-wise f32 accumulation (dispatch)
///
/// Computes `dst[k] += src[k]` for `k in 0..n` using the widest available
/// SIMD. Used for prefix-sum accumulation of histogram bins over targets.
///
/// ### Params
///
/// * `dst` - Destination slice (mutated in place).
/// * `src` - Source slice to add from.
/// * `n` - Number of elements to process.
#[inline]
pub fn accumulate_f32_simd(dst: &mut [f32], src: &[f32], n: usize) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: see `fused_mul_square_sum_simd`.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Avx512 => accumulate_f32_avx512(dst, src, n),
            SimdLevel::Avx2 => accumulate_f32_avx2(dst, src, n),
            SimdLevel::Sse => accumulate_f32_sse(dst, src, n),
            SimdLevel::Scalar => accumulate_f32_scalar(dst, src, n),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    match detect_simd_level() {
        SimdLevel::Sse => accumulate_f32_sse(dst, src, n),
        _ => accumulate_f32_scalar(dst, src, n),
    }
}

//////////////////
// Split scores //
//////////////////

/// Split score evaluation (scalar fallback)
///
/// ### Params
///
/// * `parent_vars` - Per-target parent node variance.
/// * `y_sums_total` - Per-target Y sums for the full node.
/// * `y_sum_sqs_total` - Per-target Y squared sums for the full node.
/// * `cum_y_sums` - Cumulative Y sums at the split threshold.
/// * `cum_y_sum_sqs` - Cumulative Y squared sums at the split threshold.
/// * `n_targets` - Number of active targets.
/// * `inv_nl` - 1.0 / n_left.
/// * `inv_nr` - 1.0 / n_right.
/// * `wl` - n_left / n.
/// * `wr` - n_right / n.
///
/// ### Returns
///
/// Sum of per-target weighted variance reductions.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn evaluate_split_score_f32_scalar(
    parent_vars: &[f32],
    y_sums_total: &[f32],
    y_sum_sqs_total: &[f32],
    cum_y_sums: &[f32],
    cum_y_sum_sqs: &[f32],
    n_targets: usize,
    inv_nl: f32,
    inv_nr: f32,
    wl: f32,
    wr: f32,
) -> f32 {
    let mut score = 0.0f32;
    for k in 0..n_targets {
        let y_sum_l = cum_y_sums[k];
        let y_sum_sq_l = cum_y_sum_sqs[k];
        let y_sum_r = y_sums_total[k] - y_sum_l;
        let y_sum_sq_r = y_sum_sqs_total[k] - y_sum_sq_l;
        let mean_l = y_sum_l * inv_nl;
        let var_l = f32::max(0.0, y_sum_sq_l * inv_nl - mean_l * mean_l);
        let mean_r = y_sum_r * inv_nr;
        let var_r = f32::max(0.0, y_sum_sq_r * inv_nr - mean_r * mean_r);
        score += parent_vars[k] - wl * var_l - wr * var_r;
    }
    score
}

/// Split score evaluation (128-bit: SSE2 / NEON)
///
/// ### Params
///
/// * `parent_vars` - Per-target parent node variance.
/// * `y_sums_total` - Per-target Y sums for the full node.
/// * `y_sum_sqs_total` - Per-target Y squared sums for the full node.
/// * `cum_y_sums` - Cumulative Y sums at the split threshold.
/// * `cum_y_sum_sqs` - Cumulative Y squared sums at the split threshold.
/// * `n_targets` - Number of active targets.
/// * `inv_nl` - 1.0 / n_left.
/// * `inv_nr` - 1.0 / n_right.
/// * `wl` - n_left / n.
/// * `wr` - n_right / n.
///
/// ### Returns
///
/// Sum of per-target weighted variance reductions.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn evaluate_split_score_f32_sse(
    parent_vars: &[f32],
    y_sums_total: &[f32],
    y_sum_sqs_total: &[f32],
    cum_y_sums: &[f32],
    cum_y_sum_sqs: &[f32],
    n_targets: usize,
    inv_nl: f32,
    inv_nr: f32,
    wl: f32,
    wr: f32,
) -> f32 {
    let inv_nl_v = f32x4::splat(inv_nl);
    let inv_nr_v = f32x4::splat(inv_nr);
    let wl_v = f32x4::splat(wl);
    let wr_v = f32x4::splat(wr);
    let zero_v = f32x4::ZERO;
    let chunks = n_targets / 4;
    let mut acc = f32x4::ZERO;

    unsafe {
        let pv = parent_vars.as_ptr();
        let ys = y_sums_total.as_ptr();
        let yss = y_sum_sqs_total.as_ptr();
        let cys = cum_y_sums.as_ptr();
        let cyss = cum_y_sum_sqs.as_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let parent_v = f32x4::from(*(pv.add(off) as *const [f32; 4]));
            let y_sum_l = f32x4::from(*(cys.add(off) as *const [f32; 4]));
            let y_sum_sq_l = f32x4::from(*(cyss.add(off) as *const [f32; 4]));
            let y_sum_r = f32x4::from(*(ys.add(off) as *const [f32; 4])) - y_sum_l;
            let y_sum_sq_r = f32x4::from(*(yss.add(off) as *const [f32; 4])) - y_sum_sq_l;
            let mean_l = y_sum_l * inv_nl_v;
            let var_l = (y_sum_sq_l * inv_nl_v - mean_l * mean_l).max(zero_v);
            let mean_r = y_sum_r * inv_nr_v;
            let var_r = (y_sum_sq_r * inv_nr_v - mean_r * mean_r).max(zero_v);
            acc += parent_v - wl_v * var_l - wr_v * var_r;
        }
    }

    let mut score = acc.reduce_add();
    for k in (chunks * 4)..n_targets {
        let y_sum_l = cum_y_sums[k];
        let y_sum_sq_l = cum_y_sum_sqs[k];
        let y_sum_r = y_sums_total[k] - y_sum_l;
        let y_sum_sq_r = y_sum_sqs_total[k] - y_sum_sq_l;
        let mean_l = y_sum_l * inv_nl;
        let var_l = f32::max(0.0, y_sum_sq_l * inv_nl - mean_l * mean_l);
        let mean_r = y_sum_r * inv_nr;
        let var_r = f32::max(0.0, y_sum_sq_r * inv_nr - mean_r * mean_r);
        score += parent_vars[k] - wl * var_l - wr * var_r;
    }
    score
}

/// Split score evaluation (256-bit: AVX2)
///
/// ### Params
///
/// * `parent_vars` - Per-target parent node variance.
/// * `y_sums_total` - Per-target Y sums for the full node.
/// * `y_sum_sqs_total` - Per-target Y squared sums for the full node.
/// * `cum_y_sums` - Cumulative Y sums at the split threshold (already offset
///   to h_base).
/// * `cum_y_sum_sqs` - Cumulative Y squared sums at the split threshold.
/// * `n_targets` - Number of active targets.
/// * `inv_nl` - 1.0 / n_left.
/// * `inv_nr` - 1.0 / n_right.
/// * `wl` - n_left / n.
/// * `wr` - n_right / n.
///
/// ### Returns
///
/// Sum of per-target weighted variance reductions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn evaluate_split_score_f32_avx2(
    parent_vars: &[f32],
    y_sums_total: &[f32],
    y_sum_sqs_total: &[f32],
    cum_y_sums: &[f32],
    cum_y_sum_sqs: &[f32],
    n_targets: usize,
    inv_nl: f32,
    inv_nr: f32,
    wl: f32,
    wr: f32,
) -> f32 {
    const W: usize = 8;

    let mut score = unsafe {
        let inv_nl_v = _mm256_set1_ps(inv_nl);
        let inv_nr_v = _mm256_set1_ps(inv_nr);
        let wl_v = _mm256_set1_ps(wl);
        let wr_v = _mm256_set1_ps(wr);
        let zero_v = _mm256_setzero_ps();
        let chunks = n_targets / W;
        let mut acc = _mm256_setzero_ps();

        let pv = parent_vars.as_ptr();
        let ys = y_sums_total.as_ptr();
        let yss = y_sum_sqs_total.as_ptr();
        let cys = cum_y_sums.as_ptr();
        let cyss = cum_y_sum_sqs.as_ptr();

        for i in 0..chunks {
            let off = i * W;
            let parent_v = _mm256_loadu_ps(pv.add(off));
            let y_sum_l = _mm256_loadu_ps(cys.add(off));
            let y_sum_sq_l = _mm256_loadu_ps(cyss.add(off));
            let y_sum_r = _mm256_sub_ps(_mm256_loadu_ps(ys.add(off)), y_sum_l);
            let y_sum_sq_r = _mm256_sub_ps(_mm256_loadu_ps(yss.add(off)), y_sum_sq_l);
            let mean_l = _mm256_mul_ps(y_sum_l, inv_nl_v);
            let var_l = _mm256_max_ps(
                _mm256_sub_ps(
                    _mm256_mul_ps(y_sum_sq_l, inv_nl_v),
                    _mm256_mul_ps(mean_l, mean_l),
                ),
                zero_v,
            );
            let mean_r = _mm256_mul_ps(y_sum_r, inv_nr_v);
            let var_r = _mm256_max_ps(
                _mm256_sub_ps(
                    _mm256_mul_ps(y_sum_sq_r, inv_nr_v),
                    _mm256_mul_ps(mean_r, mean_r),
                ),
                zero_v,
            );
            let term = _mm256_sub_ps(
                _mm256_sub_ps(parent_v, _mm256_mul_ps(wl_v, var_l)),
                _mm256_mul_ps(wr_v, var_r),
            );
            acc = _mm256_add_ps(acc, term);
        }

        let mut tmp = [0.0f32; W];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        tmp.iter().sum::<f32>()
    };

    for k in (n_targets / W * W)..n_targets {
        let y_sum_l = cum_y_sums[k];
        let y_sum_sq_l = cum_y_sum_sqs[k];
        let y_sum_r = y_sums_total[k] - y_sum_l;
        let y_sum_sq_r = y_sum_sqs_total[k] - y_sum_sq_l;
        let mean_l = y_sum_l * inv_nl;
        let var_l = f32::max(0.0, y_sum_sq_l * inv_nl - mean_l * mean_l);
        let mean_r = y_sum_r * inv_nr;
        let var_r = f32::max(0.0, y_sum_sq_r * inv_nr - mean_r * mean_r);
        score += parent_vars[k] - wl * var_l - wr * var_r;
    }
    score
}

/// Split score evaluation (512-bit: AVX-512F)
///
/// ### Params
///
/// * `parent_vars` - Per-target parent node variance.
/// * `y_sums_total` - Per-target Y sums for the full node.
/// * `y_sum_sqs_total` - Per-target Y squared sums for the full node.
/// * `cum_y_sums` - Cumulative Y sums at the split threshold (already offset
///   to h_base).
/// * `cum_y_sum_sqs` - Cumulative Y squared sums at the split threshold.
/// * `n_targets` - Number of active targets.
/// * `inv_nl` - 1.0 / n_left.
/// * `inv_nr` - 1.0 / n_right.
/// * `wl` - n_left / n.
/// * `wr` - n_right / n.
///
/// ### Returns
///
/// Sum of per-target weighted variance reductions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn evaluate_split_score_f32_avx512(
    parent_vars: &[f32],
    y_sums_total: &[f32],
    y_sum_sqs_total: &[f32],
    cum_y_sums: &[f32],
    cum_y_sum_sqs: &[f32],
    n_targets: usize,
    inv_nl: f32,
    inv_nr: f32,
    wl: f32,
    wr: f32,
) -> f32 {
    const W: usize = 16;

    let mut score = unsafe {
        let inv_nl_v = _mm512_set1_ps(inv_nl);
        let inv_nr_v = _mm512_set1_ps(inv_nr);
        let wl_v = _mm512_set1_ps(wl);
        let wr_v = _mm512_set1_ps(wr);
        let zero_v = _mm512_setzero_ps();
        let chunks = n_targets / W;
        let mut acc = _mm512_setzero_ps();

        let pv = parent_vars.as_ptr();
        let ys = y_sums_total.as_ptr();
        let yss = y_sum_sqs_total.as_ptr();
        let cys = cum_y_sums.as_ptr();
        let cyss = cum_y_sum_sqs.as_ptr();

        for i in 0..chunks {
            let off = i * W;
            let parent_v = _mm512_loadu_ps(pv.add(off));
            let y_sum_l = _mm512_loadu_ps(cys.add(off));
            let y_sum_sq_l = _mm512_loadu_ps(cyss.add(off));
            let y_sum_r = _mm512_sub_ps(_mm512_loadu_ps(ys.add(off)), y_sum_l);
            let y_sum_sq_r = _mm512_sub_ps(_mm512_loadu_ps(yss.add(off)), y_sum_sq_l);
            let mean_l = _mm512_mul_ps(y_sum_l, inv_nl_v);
            let var_l = _mm512_max_ps(
                _mm512_sub_ps(
                    _mm512_mul_ps(y_sum_sq_l, inv_nl_v),
                    _mm512_mul_ps(mean_l, mean_l),
                ),
                zero_v,
            );
            let mean_r = _mm512_mul_ps(y_sum_r, inv_nr_v);
            let var_r = _mm512_max_ps(
                _mm512_sub_ps(
                    _mm512_mul_ps(y_sum_sq_r, inv_nr_v),
                    _mm512_mul_ps(mean_r, mean_r),
                ),
                zero_v,
            );
            let term = _mm512_sub_ps(
                _mm512_sub_ps(parent_v, _mm512_mul_ps(wl_v, var_l)),
                _mm512_mul_ps(wr_v, var_r),
            );
            acc = _mm512_add_ps(acc, term);
        }

        _mm512_reduce_add_ps(acc)
    };

    for k in (n_targets / W * W)..n_targets {
        let y_sum_l = cum_y_sums[k];
        let y_sum_sq_l = cum_y_sum_sqs[k];
        let y_sum_r = y_sums_total[k] - y_sum_l;
        let y_sum_sq_r = y_sum_sqs_total[k] - y_sum_sq_l;
        let mean_l = y_sum_l * inv_nl;
        let var_l = f32::max(0.0, y_sum_sq_l * inv_nl - mean_l * mean_l);
        let mean_r = y_sum_r * inv_nr;
        let var_r = f32::max(0.0, y_sum_sq_r * inv_nr - mean_r * mean_r);
        score += parent_vars[k] - wl * var_l - wr * var_r;
    }
    score
}

/// Compute the total variance reduction score for a candidate split across
/// all targets using f32 arithmetic and the widest available SIMD.
///
/// ### Params
///
/// * `parent_vars` - Per-target parent node variance.
/// * `y_sums_total` - Per-target Y sums for the full node.
/// * `y_sum_sqs_total` - Per-target Y squared sums for the full node.
/// * `cum_y_sums` - Cumulative Y sums at the split threshold (already offset
///   to h_base).
/// * `cum_y_sum_sqs` - Cumulative Y squared sums at the split threshold.
/// * `n_targets` - Number of active targets.
/// * `inv_nl` - 1.0 / n_left.
/// * `inv_nr` - 1.0 / n_right.
/// * `wl` - n_left / n.
/// * `wr` - n_right / n.
///
/// ### Returns
///
/// Sum of per-target weighted variance reductions.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn evaluate_split_score_f32_simd(
    parent_vars: &[f32],
    y_sums_total: &[f32],
    y_sum_sqs_total: &[f32],
    cum_y_sums: &[f32],
    cum_y_sum_sqs: &[f32],
    n_targets: usize,
    inv_nl: f32,
    inv_nr: f32,
    wl: f32,
    wr: f32,
) -> f32 {
    // Ten arguments threaded through four arms; the macro keeps the dispatch
    // readable rather than repeating the list.
    macro_rules! call {
        ($f:ident) => {
            $f(
                parent_vars,
                y_sums_total,
                y_sum_sqs_total,
                cum_y_sums,
                cum_y_sum_sqs,
                n_targets,
                inv_nl,
                inv_nr,
                wl,
                wr,
            )
        };
    }

    #[cfg(target_arch = "x86_64")]
    // SAFETY: see `fused_mul_square_sum_simd`.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Avx512 => call!(evaluate_split_score_f32_avx512),
            SimdLevel::Avx2 => call!(evaluate_split_score_f32_avx2),
            SimdLevel::Sse => call!(evaluate_split_score_f32_sse),
            SimdLevel::Scalar => call!(evaluate_split_score_f32_scalar),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    match detect_simd_level() {
        SimdLevel::Sse => call!(evaluate_split_score_f32_sse),
        _ => call!(evaluate_split_score_f32_scalar),
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

    /// Lengths sweeping the 4-, 8- and 16-wide boundaries and the unrolled
    /// block sizes (4x8 = 32, 4x16 = 64), so both the block loop and every
    /// tail path get exercised.
    const LENGTHS: [usize; 18] = [
        1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 257, 1000,
    ];

    /// Every arm must agree with its scalar reference. Horizontal order and
    /// FMA contraction differ between widths, so this is a tolerance check.
    ///
    /// On aarch64 only the 128-bit arm runs; the x86 arms are covered wherever
    /// CI provides the hardware. The point of the x86 blocks is as much
    /// compile coverage as execution: under the old `cfg(target_feature)`
    /// gating those bodies were never built at all.
    #[test]
    fn test_sc_simd_widths_agree() {
        let mut rng = StdRng::seed_from_u64(11);

        for n in LENGTHS {
            let a: Vec<f32> = (0..n).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            let b: Vec<f32> = (0..n).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            let c: Vec<f32> = (0..n).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            // Strictly positive, so the `sqrt` in `center_values` is defined.
            let var: Vec<f32> = (0..n).map(|_| rng.random::<f32>() + 0.5).collect();

            let reference = fused_mul_square_sum_scalar(&a, &b);
            assert_relative_eq!(
                fused_mul_square_sum_sse(&a, &b),
                reference,
                epsilon = 1e-4,
                max_relative = 1e-5
            );

            let mut want = a.clone();
            center_values_scalar(&mut want, &b, &var);
            let mut got = a.clone();
            center_values_sse(&mut got, &b, &var);
            assert_eq!(got, want, "center_values sse at n {}", n);

            let mut mul_want = vec![0.0f32; n];
            elementwise_mul_scalar(&a, &b, &mut mul_want);
            let mut mul_got = vec![0.0f32; n];
            elementwise_mul_sse(&a, &b, &mut mul_got);
            assert_eq!(mul_got, mul_want, "elementwise_mul sse at n {}", n);

            let mut fma_want = vec![0.0f32; n];
            fused_mul_add_scalar(&a, &b, &c, &mut fma_want);
            let mut fma_got = vec![0.0f32; n];
            fused_mul_add_sse(&a, &b, &c, &mut fma_got);
            assert_eq!(fma_got, fma_want, "fused_mul_add sse at n {}", n);

            let mut acc_want = c.clone();
            accumulate_f32_scalar(&mut acc_want, &a, n);
            let mut acc_got = c.clone();
            accumulate_f32_sse(&mut acc_got, &a, n);
            assert_eq!(acc_got, acc_want, "accumulate sse at n {}", n);

            let score_ref =
                evaluate_split_score_f32_scalar(&a, &b, &var, &c, &var, n, 0.25, 0.5, 0.4, 0.6);
            assert_relative_eq!(
                evaluate_split_score_f32_sse(&a, &b, &var, &c, &var, n, 0.25, 0.5, 0.4, 0.6),
                score_ref,
                epsilon = 1e-3,
                max_relative = 1e-4
            );

            #[cfg(target_arch = "x86_64")]
            unsafe {
                for (has_feature, is_512) in [
                    (
                        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
                        false,
                    ),
                    (is_x86_feature_detected!("avx512f"), true),
                ] {
                    if !has_feature {
                        continue;
                    }

                    let fmss = if is_512 {
                        fused_mul_square_sum_avx512(&a, &b)
                    } else {
                        fused_mul_square_sum_avx2(&a, &b)
                    };
                    assert_relative_eq!(fmss, reference, epsilon = 1e-4, max_relative = 1e-5);

                    let mut got = a.clone();
                    if is_512 {
                        center_values_avx512(&mut got, &b, &var);
                    } else {
                        center_values_avx2(&mut got, &b, &var);
                    }
                    assert_eq!(got, want, "center_values at n {} (512 {:?})", n, is_512);

                    let mut mul_got = vec![0.0f32; n];
                    if is_512 {
                        elementwise_mul_avx512(&a, &b, &mut mul_got);
                    } else {
                        elementwise_mul_avx2(&a, &b, &mut mul_got);
                    }
                    assert_eq!(mul_got, mul_want, "elementwise_mul at n {}", n);

                    // The vector arms contract to a single fused multiply-add
                    // while the scalar reference rounds twice, so this one is
                    // approximate where the others are exact.
                    let mut fma_got = vec![0.0f32; n];
                    if is_512 {
                        fused_mul_add_avx512(&a, &b, &c, &mut fma_got);
                    } else {
                        fused_mul_add_avx2(&a, &b, &c, &mut fma_got);
                    }
                    for (got, want) in fma_got.iter().zip(fma_want.iter()) {
                        assert_relative_eq!(got, want, epsilon = 1e-6, max_relative = 1e-6);
                    }

                    let mut acc_got = c.clone();
                    if is_512 {
                        accumulate_f32_avx512(&mut acc_got, &a, n);
                    } else {
                        accumulate_f32_avx2(&mut acc_got, &a, n);
                    }
                    assert_eq!(acc_got, acc_want, "accumulate at n {}", n);

                    let score = if is_512 {
                        evaluate_split_score_f32_avx512(
                            &a, &b, &var, &c, &var, n, 0.25, 0.5, 0.4, 0.6,
                        )
                    } else {
                        evaluate_split_score_f32_avx2(
                            &a, &b, &var, &c, &var, n, 0.25, 0.5, 0.4, 0.6,
                        )
                    };
                    assert_relative_eq!(score, score_ref, epsilon = 1e-3, max_relative = 1e-4);
                }
            }
        }
    }
}
