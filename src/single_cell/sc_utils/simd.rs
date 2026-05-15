//! SIMD specifically designed for single cell applications

use wide::{f32x4, f32x8};

use crate::utils::simd::{SimdLevel, detect_simd_level};

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
#[inline(always)]
fn fused_mul_square_sum_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let mut acc = f32x8::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va = f32x8::from(*(a_ptr.add(offset) as *const [f32; 8]));
            let vb = f32x8::from(*(b_ptr.add(offset) as *const [f32; 8]));
            acc += va * vb * vb;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 8)..len {
        sum += a[i] * b[i] * b[i];
    }
    sum
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
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn fused_mul_square_sum_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 16;

    unsafe {
        let mut acc = _mm512_setzero_ps();

        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            let vb_sq = _mm512_mul_ps(vb, vb);
            acc = _mm512_fmadd_ps(va, vb_sq, acc);
        }

        let mut sum = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..len {
            sum += a[i] * b[i] * b[i];
        }
        sum
    }
}

/// SIMD-fused multiply-square-sum (512-bit fallback)
///
/// ### Params
///
/// * `a`: The first vector.
/// * `b`: The second vector.
///
/// ### Returns
///
/// The product
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn fused_mul_square_sum_avx512(a: &[f32], b: &[f32]) -> f32 {
    fused_mul_square_sum_avx2(a, b)
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
    match detect_simd_level() {
        SimdLevel::Avx512 => fused_mul_square_sum_avx512(a, b),
        SimdLevel::Avx2 => fused_mul_square_sum_avx2(a, b),
        SimdLevel::Sse => fused_mul_square_sum_sse(a, b),
        SimdLevel::Scalar => fused_mul_square_sum_scalar(a, b),
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
#[inline(always)]
fn center_values_avx2(vals: &mut [f32], mu: &[f32], var: &[f32]) {
    let len = vals.len();
    let chunks = len / 8;

    unsafe {
        let vals_ptr: *mut f32 = vals.as_mut_ptr();
        let mu_ptr: *const f32 = mu.as_ptr();
        let var_ptr: *const f32 = var.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let v = f32x8::from(*(vals_ptr.add(offset) as *const [f32; 8]));
            let m = f32x8::from(*(mu_ptr.add(offset) as *const [f32; 8]));
            let va = f32x8::from(*(var_ptr.add(offset) as *const [f32; 8]));

            let result = (v - m) / va.sqrt();
            *(vals_ptr.add(offset) as *mut [f32; 8]) = result.into();
        }
    }

    for i in (chunks * 8)..len {
        vals[i] = (vals[i] - mu[i]) / var[i].sqrt();
    }
}

/// SIMD center the values given mu and var (512-bit)
///
/// ### Params
///
/// * `vals`: The values to center.
/// * `mu`: The mean values.
/// * `var`: The variance values.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn center_values_avx512(vals: &mut [f32], mu: &[f32], var: &[f32]) {
    use std::arch::x86_64::*;

    let len = vals.len();
    let chunks = len / 16;

    unsafe {
        for i in 0..chunks {
            let offset = i * 16;
            let v = _mm512_loadu_ps(vals.as_ptr().add(offset));
            let m = _mm512_loadu_ps(mu.as_ptr().add(offset));
            let va = _mm512_loadu_ps(var.as_ptr().add(offset));

            let sqrt_va = _mm512_sqrt_ps(va);
            let diff = _mm512_sub_ps(v, m);
            let result = _mm512_div_ps(diff, sqrt_va);

            _mm512_storeu_ps(vals.as_mut_ptr().add(offset), result);
        }
    }

    for i in (chunks * 16)..len {
        vals[i] = (vals[i] - mu[i]) / var[i].sqrt();
    }
}

/// SIMD center the values given mu and var (512-bit fallback)
///
/// ### Params
///
/// * `vals`: The values to center.
/// * `mu`: The mean values.
/// * `var`: The variance values.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn center_values_avx512(vals: &mut [f32], mu: &[f32], var: &[f32]) {
    center_values_avx2(vals, mu, var)
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
    match detect_simd_level() {
        SimdLevel::Avx512 => center_values_avx512(vals, mu, var),
        SimdLevel::Avx2 => center_values_avx2(vals, mu, var),
        SimdLevel::Sse => center_values_sse(vals, mu, var),
        SimdLevel::Scalar => center_values_scalar(vals, mu, var),
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
#[inline(always)]
fn elementwise_mul_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len();
    let chunks = len / 8;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va = f32x8::from(*(a_ptr.add(offset) as *const [f32; 8]));
            let vb = f32x8::from(*(b_ptr.add(offset) as *const [f32; 8]));
            let result = va * vb;
            *(out_ptr.add(offset) as *mut [f32; 8]) = result.into();
        }
    }

    for i in (chunks * 8)..len {
        out[i] = a[i] * b[i];
    }
}

/// SIMD element-wise multiplication (512-bit)
///
/// ### Params
///
/// * `a`: The first input array.
/// * `b`: The second input array.
/// * `out`: The output array for results.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn elementwise_mul_avx512(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 16;

    unsafe {
        for i in 0..chunks {
            let offset = i * 16;
            let va = _mm512_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
            let result = _mm512_mul_ps(va, vb);
            _mm512_storeu_ps(out.as_mut_ptr().add(offset), result);
        }
    }

    for i in (chunks * 16)..len {
        out[i] = a[i] * b[i];
    }
}

/// SIMD element-wise multiplication (512-bit fallback)
///
/// ### Params
///
/// * `a`: The first input array.
/// * `b`: The second input array.
/// * `out`: The output array for results.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn elementwise_mul_avx512(a: &[f32], b: &[f32], out: &mut [f32]) {
    elementwise_mul_avx2(a, b, out)
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
    match detect_simd_level() {
        SimdLevel::Avx512 => elementwise_mul_avx512(a, b, out),
        SimdLevel::Avx2 => elementwise_mul_avx2(a, b, out),
        SimdLevel::Sse => elementwise_mul_sse(a, b, out),
        SimdLevel::Scalar => elementwise_mul_scalar(a, b, out),
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
#[inline(always)]
fn fused_mul_add_avx2(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    let len = a.len();
    let chunks = len / 8;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va = f32x8::from(*(a_ptr.add(offset) as *const [f32; 8]));
            let vb = f32x8::from(*(b_ptr.add(offset) as *const [f32; 8]));
            let vc = f32x8::from(*(c_ptr.add(offset) as *const [f32; 8]));
            let result = va * vb + vc;
            *(out_ptr.add(offset) as *mut [f32; 8]) = result.into();
        }
    }

    for i in (chunks * 8)..len {
        out[i] = a[i] * b[i] + c[i];
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
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn fused_mul_add_avx512(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 16;

    unsafe {
        for i in 0..chunks {
            let offset = i * 16;
            let va = _mm512_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
            let vc = _mm512_loadu_ps(c.as_ptr().add(offset));
            let result = _mm512_fmadd_ps(va, vb, vc);
            _mm512_storeu_ps(out.as_mut_ptr().add(offset), result);
        }
    }

    for i in (chunks * 16)..len {
        out[i] = a[i] * b[i] + c[i];
    }
}

/// SIMD fused multiply-add (512-bit fallback)
///
/// ### Params
///
/// * `a`: The first input array.
/// * `b`: The second input array.
/// * `c`: The third input array to add.
/// * `out`: The output array for results.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn fused_mul_add_avx512(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    fused_mul_add_avx2(a, b, c, out)
}

/// SIMD fused multiply-add (dispatch)
///
/// ### Params
///
/// * `a`: The first input array.
/// * `b`: The second input array.
/// * `c`: The third input array to add.
/// * `out`: The output array for results.
#[inline]
pub fn fused_mul_add_simd(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    match detect_simd_level() {
        SimdLevel::Avx512 => fused_mul_add_avx512(a, b, c, out),
        SimdLevel::Avx2 => fused_mul_add_avx2(a, b, c, out),
        SimdLevel::Sse => fused_mul_add_sse(a, b, c, out),
        SimdLevel::Scalar => fused_mul_add_scalar(a, b, c, out),
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
#[inline(always)]
fn accumulate_f32_avx2(dst: &mut [f32], src: &[f32], n: usize) {
    let chunks = n / 8;
    unsafe {
        let dst_ptr = dst.as_mut_ptr();
        let src_ptr = src.as_ptr();
        for i in 0..chunks {
            let off = i * 8;
            let vd = f32x8::from(*(dst_ptr.add(off) as *const [f32; 8]));
            let vs = f32x8::from(*(src_ptr.add(off) as *const [f32; 8]));
            *(dst_ptr.add(off) as *mut [f32; 8]) = (vd + vs).into();
        }
    }
    for k in (chunks * 8)..n {
        dst[k] += src[k];
    }
}

/// Element-wise f32 accumulation (512-bit: AVX-512F)
///
/// ### Params
///
/// * `dst` - Destination slice (mutated in place).
/// * `src` - Source slice to add from.
/// * `n` - Number of elements to process.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn accumulate_f32_avx512(dst: &mut [f32], src: &[f32], n: usize) {
    use std::arch::x86_64::*;
    let chunks = n / 16;
    unsafe {
        for i in 0..chunks {
            let off = i * 16;
            let vd = _mm512_loadu_ps(dst.as_ptr().add(off));
            let vs = _mm512_loadu_ps(src.as_ptr().add(off));
            _mm512_storeu_ps(dst.as_mut_ptr().add(off), _mm512_add_ps(vd, vs));
        }
    }
    for k in (chunks * 16)..n {
        dst[k] += src[k];
    }
}

/// Element-wise f32 accumulation (512-bit fallback)
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn accumulate_f32_avx512(dst: &mut [f32], src: &[f32], n: usize) {
    accumulate_f32_avx2(dst, src, n)
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
    match detect_simd_level() {
        SimdLevel::Avx512 => accumulate_f32_avx512(dst, src, n),
        SimdLevel::Avx2 => accumulate_f32_avx2(dst, src, n),
        SimdLevel::Sse => accumulate_f32_sse(dst, src, n),
        SimdLevel::Scalar => accumulate_f32_scalar(dst, src, n),
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
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn evaluate_split_score_f32_avx2(
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
    let inv_nl_v = f32x8::splat(inv_nl);
    let inv_nr_v = f32x8::splat(inv_nr);
    let wl_v = f32x8::splat(wl);
    let wr_v = f32x8::splat(wr);
    let zero_v = f32x8::ZERO;
    let chunks = n_targets / 8;
    let mut acc = f32x8::ZERO;

    unsafe {
        let pv = parent_vars.as_ptr();
        let ys = y_sums_total.as_ptr();
        let yss = y_sum_sqs_total.as_ptr();
        let cys = cum_y_sums.as_ptr();
        let cyss = cum_y_sum_sqs.as_ptr();

        for i in 0..chunks {
            let off = i * 8;
            let parent_v = f32x8::from(*(pv.add(off) as *const [f32; 8]));
            let y_sum_l = f32x8::from(*(cys.add(off) as *const [f32; 8]));
            let y_sum_sq_l = f32x8::from(*(cyss.add(off) as *const [f32; 8]));
            let y_sum_r = f32x8::from(*(ys.add(off) as *const [f32; 8])) - y_sum_l;
            let y_sum_sq_r = f32x8::from(*(yss.add(off) as *const [f32; 8])) - y_sum_sq_l;
            let mean_l = y_sum_l * inv_nl_v;
            let var_l = (y_sum_sq_l * inv_nl_v - mean_l * mean_l).max(zero_v);
            let mean_r = y_sum_r * inv_nr_v;
            let var_r = (y_sum_sq_r * inv_nr_v - mean_r * mean_r).max(zero_v);
            acc += parent_v - wl_v * var_l - wr_v * var_r;
        }
    }

    let mut score = acc.reduce_add();
    for k in (chunks * 8)..n_targets {
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
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn evaluate_split_score_f32_avx512(
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
    evaluate_split_score_f32_avx2(
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
    match detect_simd_level() {
        SimdLevel::Avx512 => evaluate_split_score_f32_avx512(
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
        ),
        SimdLevel::Avx2 => evaluate_split_score_f32_avx2(
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
        ),
        SimdLevel::Sse => evaluate_split_score_f32_sse(
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
        ),
        SimdLevel::Scalar => evaluate_split_score_f32_scalar(
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
        ),
    }
}
