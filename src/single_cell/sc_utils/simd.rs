//! SIMD specifically designed for single cell applications
//!
//! Same split as `crate::utils::simd`: 128-bit arms go through `wide`, which
//! is real on both x86_64 (SSE2) and aarch64 (NEON), while the 256- and
//! 512-bit arms are hand-written intrinsics behind `#[target_feature]` so they
//! survive a stock build. See that module's header for why `wide`'s wider
//! types cannot stand in.

use wide::f32x4;

use crate::utils::simd::{SimdLevel, UNROLL, detect_simd_level};

#[cfg(target_arch = "x86_64")]
use crate::utils::simd::{hsum_avx_f64, hsum_sse_f64};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
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
        vals[i] = center_one(vals[i], mu[i], var[i]);
    }
}

/// Standardise one value, mapping a zero variance to zero.
///
/// Kept as a named helper rather than inlined because all four dispatch arms
/// need the identical scalar tail, and the SIMD arms mirror it with a select.
///
/// A zero variance means the model predicts the observation exactly, so the
/// standardised value is zero by definition rather than `0/0`. Without this the
/// `NaN` propagates into the autocorrelation statistic and the gene is silently
/// dropped; a single zero-depth cell would take every gene with it.
///
/// ### Params
///
/// * `val`: The value to centre.
/// * `mu`: The mean.
/// * `var`: The variance.
///
/// ### Returns
///
/// `(val - mu) / sqrt(var)`, or zero when `var` is zero.
#[inline(always)]
fn center_one(val: f32, mu: f32, var: f32) -> f32 {
    if var > 0.0 {
        (val - mu) / var.sqrt()
    } else {
        0.0
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

            // zero variance -> zero, see `center_one`
            let result = (v - m) / va.sqrt();
            let result = va.simd_gt(f32x4::ZERO).select(result, f32x4::ZERO);
            *(vals_ptr.add(offset) as *mut [f32; 4]) = result.into();
        }
    }

    for i in (chunks * 4)..len {
        vals[i] = center_one(vals[i], mu[i], var[i]);
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
            // zero variance -> zero, see `center_one`
            let raw = _mm256_div_ps(_mm256_sub_ps(v, m), _mm256_sqrt_ps(va));
            let keep = _mm256_cmp_ps(va, _mm256_setzero_ps(), _CMP_GT_OQ);
            let result = _mm256_and_ps(raw, keep);
            _mm256_storeu_ps(vals_ptr.add(off), result);
        }

        for i in (chunks * W)..len {
            vals[i] = center_one(vals[i], mu[i], var[i]);
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
            // zero variance -> zero, see `center_one`
            let raw = _mm512_div_ps(_mm512_sub_ps(v, m), _mm512_sqrt_ps(va));
            let keep = _mm512_cmp_ps_mask(va, _mm512_setzero_ps(), _CMP_GT_OQ);
            let result = _mm512_maskz_mov_ps(keep, raw);
            _mm512_storeu_ps(vals_ptr.add(off), result);
        }

        for i in (chunks * W)..len {
            vals[i] = center_one(vals[i], mu[i], var[i]);
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

///////////////
// CellSweep //
///////////////

//////////////////////////
// Vectorised f64 `ln`  //
//////////////////////////

/// Numerator coefficients of the Cephes rational approximation to `ln`.
///
/// Lifted verbatim from `wide`'s `f64x2::ln`, which is the only vector `ln` any
/// of the four arms could otherwise share: there is no `ln` intrinsic at any
/// width, and `wide`'s wider types degrade to stacked 128-bit halves in a stock
/// build, so the 256- and 512-bit arms have to carry the polynomial themselves.
///
/// Kept at the published digit count rather than truncated to what `f64` can
/// hold. The values round to the same bits either way, and leaving them
/// verbatim is what makes the four arms checkable against the reference by eye.
#[allow(clippy::excessive_precision)]
const LN_P: [f64; 6] = [
    7.70838733755885391666E0,
    1.79368678507819816313E1,
    1.44989225341610930846E1,
    4.70579119878881725854E0,
    4.97494994976747001425E-1,
    1.01875663804580931796E-4,
];

/// Denominator coefficients. Monic, so the leading `x^5` is implicit. See
/// [LN_P] for why the extra digits stay.
#[allow(clippy::excessive_precision)]
const LN_Q: [f64; 5] = [
    2.31251620126765340583E1,
    7.11544750618563894466E1,
    8.29875266912776603211E1,
    4.52279145837532221105E1,
    1.12873587189167450590E1,
];

/// High half of `ln(2)`, from fdlibm. Exact in `f64`.
const LN2_HI: f64 = f64::from_bits(0x3FE6_2E42_FEE0_0000);

/// Low half of `ln(2)`. The pair reconstructs `ln(2)` to about 106 bits, which
/// is what keeps the `exponent * ln(2)` term from dominating the error.
const LN2_LO: f64 = f64::from_bits(0x3DEA_39EF_3579_3C76);

/// `sqrt(2) / 2`. Mantissas at or below this are folded up one octave so the
/// polynomial argument stays near zero.
const LN_SQRT2_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// `2^52`. Its mantissa field is zero, which is what turns the exponent
/// extraction into a shift, a bit-or and a subtract. The usual stand-in for
/// `i64 -> f64`, which has no AVX2 instruction.
const LN_POW2_52: f64 = 4503599627370496.0;

/// Mantissa field mask.
const LN_MANTISSA_MASK: u64 = 0x000F_FFFF_FFFF_FFFF;

/// Exponent field of `0.5`, i.e. what the mantissa is rebased onto.
const LN_HALF_EXPONENT: u64 = 0x3FE0_0000_0000_0000;

/// Smallest argument the vector `ln` arms accept.
///
/// The Cephes body has no subnormal path: `fraction_2` on a subnormal reads a
/// zero exponent field and returns nonsense rather than a small negative
/// result. Every arm clamps to this before the polynomial, so the argument is
/// always a positive normal and the whole special-case ladder `wide` carries
/// around its own `ln` is unreachable here.
#[inline(always)]
fn ln_floor(log_eps: f64) -> f64 {
    log_eps.max(f64::MIN_POSITIVE)
}

/// Vectorised natural log of two `f64` lanes (128-bit, NEON).
///
/// ### Params
///
/// * `x1` - Lanes to take the log of. Must be positive, normal and finite.
///
/// ### Returns
///
/// `ln(x1)`, lane-wise.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn ln_neon_pd(x1: float64x2_t) -> float64x2_t {
    let bits = vreinterpretq_u64_f64(x1);

    // mantissa rebased onto [0.5, 1)
    let frac = vreinterpretq_f64_u64(vorrq_u64(
        vandq_u64(bits, vdupq_n_u64(LN_MANTISSA_MASK)),
        vdupq_n_u64(LN_HALF_EXPONENT),
    ));

    // unbiased exponent as f64, via the 2^52 mantissa trick
    let e = vsubq_f64(
        vreinterpretq_f64_u64(vorrq_u64(
            vshrq_n_u64(bits, 52),
            vreinterpretq_u64_f64(vdupq_n_f64(LN_POW2_52)),
        )),
        vdupq_n_f64(LN_POW2_52 + 1023.0),
    );

    let keep = vcgtq_f64(frac, vdupq_n_f64(LN_SQRT2_HALF));
    let x = vbslq_f64(keep, frac, vaddq_f64(frac, frac));
    let fe = vbslq_f64(keep, vaddq_f64(e, vdupq_n_f64(1.0)), e);

    let x = vsubq_f64(x, vdupq_n_f64(1.0));
    let x2 = vmulq_f64(x, x);

    let mut px = vdupq_n_f64(LN_P[5]);
    for c in LN_P.iter().rev().skip(1) {
        px = vfmaq_f64(vdupq_n_f64(*c), px, x);
    }
    let px = vmulq_f64(vmulq_f64(x2, x), px);

    let mut qx = vaddq_f64(x, vdupq_n_f64(LN_Q[4]));
    for c in LN_Q.iter().rev().skip(1) {
        qx = vfmaq_f64(vdupq_n_f64(*c), qx, x);
    }

    let res = vdivq_f64(px, qx);
    let res = vfmaq_f64(res, fe, vdupq_n_f64(LN2_LO));
    let res = vaddq_f64(res, vfmsq_f64(x, x2, vdupq_n_f64(0.5)));
    vfmaq_f64(res, fe, vdupq_n_f64(LN2_HI))
}

/// Vectorised natural log of two `f64` lanes (128-bit, SSE4.1).
///
/// SSE2 has no FMA, so the polynomial is a multiply and an add per step. That
/// costs an extra rounding per Horner step against the wider arms, which is
/// well inside the tolerance the callers need.
///
/// ### Params
///
/// * `x1` - Lanes to take the log of. Must be positive, normal and finite.
///
/// ### Returns
///
/// `ln(x1)`, lane-wise.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn ln_sse_pd(x1: __m128d) -> __m128d {
    let bits = _mm_castpd_si128(x1);

    let frac = _mm_castsi128_pd(_mm_or_si128(
        _mm_and_si128(bits, _mm_set1_epi64x(LN_MANTISSA_MASK as i64)),
        _mm_set1_epi64x(LN_HALF_EXPONENT as i64),
    ));

    let e = _mm_sub_pd(
        _mm_castsi128_pd(_mm_or_si128(
            _mm_srli_epi64(bits, 52),
            _mm_castpd_si128(_mm_set1_pd(LN_POW2_52)),
        )),
        _mm_set1_pd(LN_POW2_52 + 1023.0),
    );

    let keep = _mm_cmpgt_pd(frac, _mm_set1_pd(LN_SQRT2_HALF));
    let x = _mm_blendv_pd(_mm_add_pd(frac, frac), frac, keep);
    let fe = _mm_blendv_pd(e, _mm_add_pd(e, _mm_set1_pd(1.0)), keep);

    let x = _mm_sub_pd(x, _mm_set1_pd(1.0));
    let x2 = _mm_mul_pd(x, x);

    let mut px = _mm_set1_pd(LN_P[5]);
    for c in LN_P.iter().rev().skip(1) {
        px = _mm_add_pd(_mm_mul_pd(px, x), _mm_set1_pd(*c));
    }
    let px = _mm_mul_pd(_mm_mul_pd(x2, x), px);

    let mut qx = _mm_add_pd(x, _mm_set1_pd(LN_Q[4]));
    for c in LN_Q.iter().rev().skip(1) {
        qx = _mm_add_pd(_mm_mul_pd(qx, x), _mm_set1_pd(*c));
    }

    let res = _mm_div_pd(px, qx);
    let res = _mm_add_pd(res, _mm_mul_pd(fe, _mm_set1_pd(LN2_LO)));
    let res = _mm_add_pd(res, _mm_sub_pd(x, _mm_mul_pd(x2, _mm_set1_pd(0.5))));
    _mm_add_pd(res, _mm_mul_pd(fe, _mm_set1_pd(LN2_HI)))
}

/// Vectorised natural log of four `f64` lanes (256-bit).
///
/// ### Params
///
/// * `x1` - Lanes to take the log of. Must be positive, normal and finite.
///
/// ### Returns
///
/// `ln(x1)`, lane-wise.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn ln_avx2_pd(x1: __m256d) -> __m256d {
    let bits = _mm256_castpd_si256(x1);

    let frac = _mm256_castsi256_pd(_mm256_or_si256(
        _mm256_and_si256(bits, _mm256_set1_epi64x(LN_MANTISSA_MASK as i64)),
        _mm256_set1_epi64x(LN_HALF_EXPONENT as i64),
    ));

    let e = _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_or_si256(
            _mm256_srli_epi64(bits, 52),
            _mm256_castpd_si256(_mm256_set1_pd(LN_POW2_52)),
        )),
        _mm256_set1_pd(LN_POW2_52 + 1023.0),
    );

    let keep = _mm256_cmp_pd(frac, _mm256_set1_pd(LN_SQRT2_HALF), _CMP_GT_OQ);
    let x = _mm256_blendv_pd(_mm256_add_pd(frac, frac), frac, keep);
    let fe = _mm256_blendv_pd(e, _mm256_add_pd(e, _mm256_set1_pd(1.0)), keep);

    let x = _mm256_sub_pd(x, _mm256_set1_pd(1.0));
    let x2 = _mm256_mul_pd(x, x);

    let mut px = _mm256_set1_pd(LN_P[5]);
    for c in LN_P.iter().rev().skip(1) {
        px = _mm256_fmadd_pd(px, x, _mm256_set1_pd(*c));
    }
    let px = _mm256_mul_pd(_mm256_mul_pd(x2, x), px);

    let mut qx = _mm256_add_pd(x, _mm256_set1_pd(LN_Q[4]));
    for c in LN_Q.iter().rev().skip(1) {
        qx = _mm256_fmadd_pd(qx, x, _mm256_set1_pd(*c));
    }

    let res = _mm256_div_pd(px, qx);
    let res = _mm256_fmadd_pd(fe, _mm256_set1_pd(LN2_LO), res);
    let res = _mm256_add_pd(res, _mm256_fnmadd_pd(x2, _mm256_set1_pd(0.5), x));
    _mm256_fmadd_pd(fe, _mm256_set1_pd(LN2_HI), res)
}

/// Vectorised natural log of eight `f64` lanes (512-bit).
///
/// Uses the same shift-and-or exponent extraction as the narrower arms rather
/// than `_mm512_getexp_pd` / `_mm512_getmant_pd`, so all four bodies read the
/// same and a fix to one transfers by eye.
///
/// ### Params
///
/// * `x1` - Lanes to take the log of. Must be positive, normal and finite.
///
/// ### Returns
///
/// `ln(x1)`, lane-wise.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn ln_avx512_pd(x1: __m512d) -> __m512d {
    let bits = _mm512_castpd_si512(x1);

    let frac = _mm512_castsi512_pd(_mm512_or_si512(
        _mm512_and_si512(bits, _mm512_set1_epi64(LN_MANTISSA_MASK as i64)),
        _mm512_set1_epi64(LN_HALF_EXPONENT as i64),
    ));

    let e = _mm512_sub_pd(
        _mm512_castsi512_pd(_mm512_or_si512(
            _mm512_srli_epi64(bits, 52),
            _mm512_castpd_si512(_mm512_set1_pd(LN_POW2_52)),
        )),
        _mm512_set1_pd(LN_POW2_52 + 1023.0),
    );

    let keep = _mm512_cmp_pd_mask(frac, _mm512_set1_pd(LN_SQRT2_HALF), _CMP_GT_OQ);
    let x = _mm512_mask_blend_pd(keep, _mm512_add_pd(frac, frac), frac);
    let fe = _mm512_mask_blend_pd(keep, e, _mm512_add_pd(e, _mm512_set1_pd(1.0)));

    let x = _mm512_sub_pd(x, _mm512_set1_pd(1.0));
    let x2 = _mm512_mul_pd(x, x);

    let mut px = _mm512_set1_pd(LN_P[5]);
    for c in LN_P.iter().rev().skip(1) {
        px = _mm512_fmadd_pd(px, x, _mm512_set1_pd(*c));
    }
    let px = _mm512_mul_pd(_mm512_mul_pd(x2, x), px);

    let mut qx = _mm512_add_pd(x, _mm512_set1_pd(LN_Q[4]));
    for c in LN_Q.iter().rev().skip(1) {
        qx = _mm512_fmadd_pd(qx, x, _mm512_set1_pd(*c));
    }

    let res = _mm512_div_pd(px, qx);
    let res = _mm512_fmadd_pd(fe, _mm512_set1_pd(LN2_LO), res);
    let res = _mm512_add_pd(res, _mm512_fnmadd_pd(x2, _mm512_set1_pd(0.5), x));
    _mm512_fmadd_pd(fe, _mm512_set1_pd(LN2_HI), res)
}

///////////////////
// Weighted logs //
///////////////////

/// Count-weighted log-likelihood over one run of non-zeros (scalar)
///
/// ### Params
///
/// * `counts` - Observed counts.
/// * `p` - Mixture probability per count, same length as `counts`.
/// * `log_eps` - Floor on the argument of `ln`.
///
/// ### Returns
///
/// `sum_j counts[j] * ln(max(p[j], log_eps))`.
#[inline(always)]
fn ln_dot_scalar(counts: &[f64], p: &[f64], log_eps: f64) -> f64 {
    let floor = ln_floor(log_eps);
    counts
        .iter()
        .zip(p.iter())
        .map(|(&v, &p)| v * p.max(floor).ln())
        .sum()
}

/// Count-weighted log-likelihood over one run of non-zeros (128-bit, NEON)
///
/// ### Params
///
/// * `counts` - Observed counts.
/// * `p` - Mixture probability per count, same length as `counts`.
/// * `log_eps` - Floor on the argument of `ln`.
///
/// ### Returns
///
/// `sum_j counts[j] * ln(max(p[j], log_eps))`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn ln_dot_neon(counts: &[f64], p: &[f64], log_eps: f64) -> f64 {
    const W: usize = 2;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = counts.len();
        let v_ptr = counts.as_ptr();
        let p_ptr = p.as_ptr();
        let floor = vdupq_n_f64(ln_floor(log_eps));
        let mut acc = [vdupq_n_f64(0.0); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                let off = base + u * W;
                let vp = vmaxq_f64(vld1q_f64(p_ptr.add(off)), floor);
                *acc = vfmaq_f64(*acc, vld1q_f64(v_ptr.add(off)), ln_neon_pd(vp));
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = vaddq_f64(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let vp = vmaxq_f64(vld1q_f64(p_ptr.add(i)), floor);
            total = vfmaq_f64(total, vld1q_f64(v_ptr.add(i)), ln_neon_pd(vp));
            i += W;
        }

        let mut sum = vaddvq_f64(total);
        let floor = ln_floor(log_eps);
        while i < len {
            sum += counts[i] * p[i].max(floor).ln();
            i += 1;
        }
        sum
    }
}

/// Count-weighted log-likelihood over one run of non-zeros (128-bit, SSE4.1)
///
/// ### Params
///
/// * `counts` - Observed counts.
/// * `p` - Mixture probability per count, same length as `counts`.
/// * `log_eps` - Floor on the argument of `ln`.
///
/// ### Returns
///
/// `sum_j counts[j] * ln(max(p[j], log_eps))`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn ln_dot_sse(counts: &[f64], p: &[f64], log_eps: f64) -> f64 {
    const W: usize = 2;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = counts.len();
        let v_ptr = counts.as_ptr();
        let p_ptr = p.as_ptr();
        let floor_v = _mm_set1_pd(ln_floor(log_eps));
        let mut acc = [_mm_setzero_pd(); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                let off = base + u * W;
                let vp = _mm_max_pd(_mm_loadu_pd(p_ptr.add(off)), floor_v);
                let term = _mm_mul_pd(_mm_loadu_pd(v_ptr.add(off)), ln_sse_pd(vp));
                *acc = _mm_add_pd(*acc, term);
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = _mm_add_pd(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let vp = _mm_max_pd(_mm_loadu_pd(p_ptr.add(i)), floor_v);
            let term = _mm_mul_pd(_mm_loadu_pd(v_ptr.add(i)), ln_sse_pd(vp));
            total = _mm_add_pd(total, term);
            i += W;
        }

        let mut sum = hsum_sse_f64(total);
        let floor = ln_floor(log_eps);
        while i < len {
            sum += counts[i] * p[i].max(floor).ln();
            i += 1;
        }
        sum
    }
}

/// Count-weighted log-likelihood over one run of non-zeros (256-bit)
///
/// ### Params
///
/// * `counts` - Observed counts.
/// * `p` - Mixture probability per count, same length as `counts`.
/// * `log_eps` - Floor on the argument of `ln`.
///
/// ### Returns
///
/// `sum_j counts[j] * ln(max(p[j], log_eps))`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn ln_dot_avx2(counts: &[f64], p: &[f64], log_eps: f64) -> f64 {
    const W: usize = 4;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = counts.len();
        let v_ptr = counts.as_ptr();
        let p_ptr = p.as_ptr();
        let floor_v = _mm256_set1_pd(ln_floor(log_eps));
        let mut acc = [_mm256_setzero_pd(); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                let off = base + u * W;
                let vp = _mm256_max_pd(_mm256_loadu_pd(p_ptr.add(off)), floor_v);
                *acc = _mm256_fmadd_pd(_mm256_loadu_pd(v_ptr.add(off)), ln_avx2_pd(vp), *acc);
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = _mm256_add_pd(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let vp = _mm256_max_pd(_mm256_loadu_pd(p_ptr.add(i)), floor_v);
            total = _mm256_fmadd_pd(_mm256_loadu_pd(v_ptr.add(i)), ln_avx2_pd(vp), total);
            i += W;
        }

        let mut sum = hsum_avx_f64(total);
        let floor = ln_floor(log_eps);
        while i < len {
            sum += counts[i] * p[i].max(floor).ln();
            i += 1;
        }
        sum
    }
}

/// Count-weighted log-likelihood over one run of non-zeros (512-bit)
///
/// ### Params
///
/// * `counts` - Observed counts.
/// * `p` - Mixture probability per count, same length as `counts`.
/// * `log_eps` - Floor on the argument of `ln`.
///
/// ### Returns
///
/// `sum_j counts[j] * ln(max(p[j], log_eps))`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn ln_dot_avx512(counts: &[f64], p: &[f64], log_eps: f64) -> f64 {
    const W: usize = 8;
    const BLOCK: usize = W * UNROLL;

    unsafe {
        let len = counts.len();
        let v_ptr = counts.as_ptr();
        let p_ptr = p.as_ptr();
        let floor_v = _mm512_set1_pd(ln_floor(log_eps));
        let mut acc = [_mm512_setzero_pd(); UNROLL];

        let n_blocks = len / BLOCK;
        for i in 0..n_blocks {
            let base = i * BLOCK;
            for (u, acc) in acc.iter_mut().enumerate() {
                let off = base + u * W;
                let vp = _mm512_max_pd(_mm512_loadu_pd(p_ptr.add(off)), floor_v);
                *acc = _mm512_fmadd_pd(_mm512_loadu_pd(v_ptr.add(off)), ln_avx512_pd(vp), *acc);
            }
        }

        let mut total = acc[0];
        for acc in &acc[1..] {
            total = _mm512_add_pd(total, *acc);
        }

        let mut i = n_blocks * BLOCK;
        while i + W <= len {
            let vp = _mm512_max_pd(_mm512_loadu_pd(p_ptr.add(i)), floor_v);
            total = _mm512_fmadd_pd(_mm512_loadu_pd(v_ptr.add(i)), ln_avx512_pd(vp), total);
            i += W;
        }

        let mut sum = _mm512_reduce_add_pd(total);
        let floor = ln_floor(log_eps);
        while i < len {
            sum += counts[i] * p[i].max(floor).ln();
            i += 1;
        }
        sum
    }
}

/// Count-weighted log-likelihood over one run of non-zeros (dispatch)
///
/// Computes `sum_j counts[j] * ln(max(p[j], log_eps))`, the per-barcode
/// log-likelihood of the CellSweep mixture. The scalar form of this loop cannot
/// vectorise: `ln` is a libm call and the `f64` reduction around it is not
/// associative, so LLVM will neither inline a vector log nor reorder the sum.
///
/// The vector arms are the Cephes rational approximation, matching `wide`'s
/// `f64x2::ln`, but with every special case stripped. That is safe only because
/// of the clamp: the argument is floored at `max(log_eps, f64::MIN_POSITIVE)`
/// before the polynomial, so it is always a positive normal. Two consequences
/// for anyone else calling this:
///
/// - A non-finite `p[j]` is not handled. `+inf` returns a large finite number
///   rather than `+inf`, and `NaN` returns `ln(floor)`. The CellSweep caller
///   errors on a non-finite log-likelihood one level up, so neither is
///   reachable there.
/// - The scalar arm goes through libm and is correctly rounded; the vector arms
///   carry a few ULP of polynomial error, and the 128-bit x86 arm a little more
///   again because SSE2 has no FMA. The arms are not bit-identical, in the same
///   way `fused_mul_add_simd`'s are not.
///
/// ### Params
///
/// * `counts` - Observed counts.
/// * `p` - Mixture probability per count, same length as `counts`.
/// * `log_eps` - Floor on the argument of `ln`.
///
/// ### Returns
///
/// The count-weighted log-likelihood.
#[inline]
pub fn ln_dot_simd(counts: &[f64], p: &[f64], log_eps: f64) -> f64 {
    debug_assert_eq!(counts.len(), p.len(), "slices must match in length");

    #[cfg(target_arch = "x86_64")]
    // SAFETY: see `fused_mul_square_sum_simd`.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Avx512 => ln_dot_avx512(counts, p, log_eps),
            SimdLevel::Avx2 => ln_dot_avx2(counts, p, log_eps),
            SimdLevel::Sse => ln_dot_sse(counts, p, log_eps),
            SimdLevel::Scalar => ln_dot_scalar(counts, p, log_eps),
        }
    }

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64, so `detect_simd_level` always
    // reports `Sse` here and the feature is unconditionally present.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Sse => ln_dot_neon(counts, p, log_eps),
            _ => ln_dot_scalar(counts, p, log_eps),
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    ln_dot_scalar(counts, p, log_eps)
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

    /// Mixture probabilities the CellSweep E-step actually produces, spanning
    /// the whole range between the `log_eps` floor and one.
    ///
    /// Log-spaced rather than uniform: the Cephes body splits on the mantissa
    /// and adds `exponent * ln(2)`, so an error in the exponent path only shows
    /// up if the sweep visits many different exponents.
    ///
    /// ### Returns
    ///
    /// `n` values log-spaced across `[1e-300, 1.0]`.
    fn log_spaced(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1).max(1) as f64;
                10.0_f64.powf(-300.0 * (1.0 - t))
            })
            .collect()
    }

    /// The vector `ln` has to match libm per element, not just in aggregate.
    ///
    /// Each probe runs eight identical lanes, which lands in the vector body on
    /// every arm: exactly one unrolled block at 128 bits, one whole vector at
    /// 256 and 512. Dividing the result back out isolates the single `ln`, so a
    /// mis-transliterated coefficient cannot hide inside the reduction.
    #[test]
    fn ln_dot_matches_libm_per_element() {
        let counts = [1.0_f64; 8];

        for x in log_spaced(600) {
            let got = ln_dot_simd(&counts, &[x; 8], 0.0) / 8.0;
            assert_relative_eq!(got, x.ln(), max_relative = 1e-14);
        }
    }

    /// Block loop, leftover whole vectors and scalar tail all have to agree
    /// with the scalar arm, at every length that splits between them.
    #[test]
    fn ln_dot_widths_agree() {
        let mut rng = StdRng::seed_from_u64(29);

        for n in LENGTHS {
            let counts: Vec<f64> = (0..n).map(|_| rng.random_range(1.0..500.0)).collect();
            let p: Vec<f64> = (0..n)
                .map(|_| 10.0_f64.powf(rng.random_range(-12.0..0.0)))
                .collect();

            let want = ln_dot_scalar(&counts, &p, 1e-300);
            assert_relative_eq!(
                ln_dot_simd(&counts, &p, 1e-300),
                want,
                max_relative = 1e-13,
                epsilon = 1e-12
            );

            // The clamp is what makes the stripped special-case ladder sound,
            // so it gets swept at every length too.
            let tiny = vec![0.0_f64; n];
            assert_relative_eq!(
                ln_dot_simd(&counts, &tiny, 1e-30),
                counts.iter().sum::<f64>() * 1e-30_f64.ln(),
                max_relative = 1e-13
            );
        }
    }

    /// Anything at or below the floor reads as the floor, and nothing returns
    /// `-inf` or `NaN`.
    ///
    /// A zero `p` is reachable in the E-step: an all-zero profile entry against
    /// a frozen ambient profile that also happens to be zero for that gene. The
    /// scalar implementation leant on `f64::ln` handling it; the vector arms
    /// have no subnormal path at all, so the floor has to do the work.
    #[test]
    fn ln_dot_floors_degenerate_probabilities() {
        let counts = vec![2.0_f64; 37];

        // Below `f64::MIN_POSITIVE` the floor itself is clamped, otherwise the
        // Cephes body would read a zero exponent field and return nonsense.
        for &log_eps in &[0.0, f64::MIN_POSITIVE / 4.0, 1e-300, 1e-12] {
            let floor = log_eps.max(f64::MIN_POSITIVE);
            let p = vec![0.0_f64; 37];
            let got = ln_dot_simd(&counts, &p, log_eps);

            assert!(got.is_finite(), "log_eps {log_eps:e} gave {got}");
            assert_relative_eq!(got, 2.0 * 37.0 * floor.ln(), max_relative = 1e-13);
        }
    }

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

    /// A zero variance standardises to zero rather than `NaN`, on every dispatch
    /// arm. Upstream `hotspot/utils.py:11-14` does the same, and without it a
    /// single zero-depth cell drops every gene from a Hotspot run.
    #[test]
    fn center_values_maps_zero_variance_to_zero() {
        // long enough to cover the widest vector body plus a scalar tail
        let n = 37;
        let mut vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mu: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let var: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { 0.0 } else { 4.0 }).collect();

        center_values_simd(&mut vals, &mu, &var);

        assert!(vals.iter().all(|v| v.is_finite()), "{vals:?}");
        // vals == mu everywhere, so every entry standardises to zero
        assert!(vals.iter().all(|v| *v == 0.0), "{vals:?}");
    }

    /// The guard must not disturb the ordinary path.
    #[test]
    fn center_values_matches_the_scalar_formula() {
        let n = 37;
        let vals_in: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 1.0).collect();
        let mu: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25).collect();
        let var: Vec<f32> = (0..n).map(|i| 1.0 + (i % 5) as f32).collect();

        let mut actual = vals_in.clone();
        center_values_simd(&mut actual, &mu, &var);

        for i in 0..n {
            let expected = (vals_in[i] - mu[i]) / var[i].sqrt();
            assert_relative_eq!(actual[i], expected, epsilon = 1e-6);
        }
    }
}
