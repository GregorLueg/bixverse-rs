//! SIMD specifically designed for single cell applications

use wide::{f32x4, f32x8};

use crate::utils::simd::{SimdLevel, detect_simd_level};

////////////
// SCENIC //
////////////

//////////////////
// Accumulation //
//////////////////

/// Element-wise f32 accumulation (scalar fallback)
#[inline(always)]
fn accumulate_f32_scalar(dst: &mut [f32], src: &[f32], n: usize) {
    for k in 0..n {
        dst[k] += src[k];
    }
}

/// Element-wise f32 accumulation (128-bit: SSE2 / NEON)
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

/// Split score evaluation (512-bit: AVX-512F)
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

/// Split score evaluation (512-bit fallback)
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
