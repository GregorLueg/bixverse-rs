//! SIMD specifically designed for single cell applications

use wide::{f64x2, f64x4};

use crate::utils::simd::{SimdLevel, detect_simd_level};

////////////
// SCENIC //
////////////

//////////////////
// Accumulation //
//////////////////

/// Element-wise f64 accumulation (scalar fallback)
#[inline(always)]
fn accumulate_f64_scalar(dst: &mut [f64], src: &[f64], n: usize) {
    for k in 0..n {
        dst[k] += src[k];
    }
}

/// Element-wise f64 accumulation (128-bit: SSE2 / NEON)
#[inline(always)]
fn accumulate_f64_sse(dst: &mut [f64], src: &[f64], n: usize) {
    let chunks = n / 2;
    unsafe {
        let dst_ptr = dst.as_mut_ptr();
        let src_ptr = src.as_ptr();
        for i in 0..chunks {
            let off = i * 2;
            let vd = f64x2::from(*(dst_ptr.add(off) as *const [f64; 2]));
            let vs = f64x2::from(*(src_ptr.add(off) as *const [f64; 2]));
            *(dst_ptr.add(off) as *mut [f64; 2]) = (vd + vs).into();
        }
    }
    for k in (chunks * 2)..n {
        dst[k] += src[k];
    }
}

/// Element-wise f64 accumulation (256-bit: AVX2)
#[inline(always)]
fn accumulate_f64_avx2(dst: &mut [f64], src: &[f64], n: usize) {
    let chunks = n / 4;
    unsafe {
        let dst_ptr = dst.as_mut_ptr();
        let src_ptr = src.as_ptr();
        for i in 0..chunks {
            let off = i * 4;
            let vd = f64x4::from(*(dst_ptr.add(off) as *const [f64; 4]));
            let vs = f64x4::from(*(src_ptr.add(off) as *const [f64; 4]));
            *(dst_ptr.add(off) as *mut [f64; 4]) = (vd + vs).into();
        }
    }
    for k in (chunks * 4)..n {
        dst[k] += src[k];
    }
}

/// Element-wise f64 accumulation (512-bit: AVX-512F)
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn accumulate_f64_avx512(dst: &mut [f64], src: &[f64], n: usize) {
    use std::arch::x86_64::*;
    let chunks = n / 8;
    unsafe {
        for i in 0..chunks {
            let off = i * 8;
            let vd = _mm512_loadu_pd(dst.as_ptr().add(off));
            let vs = _mm512_loadu_pd(src.as_ptr().add(off));
            _mm512_storeu_pd(dst.as_mut_ptr().add(off), _mm512_add_pd(vd, vs));
        }
    }
    for k in (chunks * 8)..n {
        dst[k] += src[k];
    }
}

/// Element-wise f64 accumulation (512-bit fallback)
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn accumulate_f64_avx512(dst: &mut [f64], src: &[f64], n: usize) {
    accumulate_f64_avx2(dst, src, n)
}

/// Element-wise f64 accumulation (dispatch)
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
pub fn accumulate_f64_simd(dst: &mut [f64], src: &[f64], n: usize) {
    match detect_simd_level() {
        SimdLevel::Avx512 => accumulate_f64_avx512(dst, src, n),
        SimdLevel::Avx2 => accumulate_f64_avx2(dst, src, n),
        SimdLevel::Sse => accumulate_f64_sse(dst, src, n),
        SimdLevel::Scalar => accumulate_f64_scalar(dst, src, n),
    }
}

/// Split score evaluation (scalar fallback)
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn evaluate_split_score_scalar(
    parent_vars: &[f64],
    y_sums_total: &[f64],
    y_sum_sqs_total: &[f64],
    cum_y_sums: &[f64],
    cum_y_sum_sqs: &[f64],
    n_targets: usize,
    inv_nl: f64,
    inv_nr: f64,
    wl: f64,
    wr: f64,
) -> f64 {
    let mut score = 0.0f64;
    for k in 0..n_targets {
        let y_sum_l = cum_y_sums[k];
        let y_sum_sq_l = cum_y_sum_sqs[k];
        let y_sum_r = y_sums_total[k] - y_sum_l;
        let y_sum_sq_r = y_sum_sqs_total[k] - y_sum_sq_l;

        let mean_l = y_sum_l * inv_nl;
        let var_l = f64::max(0.0, y_sum_sq_l * inv_nl - mean_l * mean_l);

        let mean_r = y_sum_r * inv_nr;
        let var_r = f64::max(0.0, y_sum_sq_r * inv_nr - mean_r * mean_r);

        score += parent_vars[k] - wl * var_l - wr * var_r;
    }
    score
}

/// Split score evaluation (128-bit: SSE2 / NEON)
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn evaluate_split_score_sse(
    parent_vars: &[f64],
    y_sums_total: &[f64],
    y_sum_sqs_total: &[f64],
    cum_y_sums: &[f64],
    cum_y_sum_sqs: &[f64],
    n_targets: usize,
    inv_nl: f64,
    inv_nr: f64,
    wl: f64,
    wr: f64,
) -> f64 {
    let inv_nl_v = f64x2::splat(inv_nl);
    let inv_nr_v = f64x2::splat(inv_nr);
    let wl_v = f64x2::splat(wl);
    let wr_v = f64x2::splat(wr);
    let zero_v = f64x2::ZERO;

    let chunks = n_targets / 2;
    let mut acc = f64x2::ZERO;

    unsafe {
        let pv = parent_vars.as_ptr();
        let ys = y_sums_total.as_ptr();
        let yss = y_sum_sqs_total.as_ptr();
        let cys = cum_y_sums.as_ptr();
        let cyss = cum_y_sum_sqs.as_ptr();

        for i in 0..chunks {
            let off = i * 2;

            let parent_v = f64x2::from(*(pv.add(off) as *const [f64; 2]));
            let y_sum_l = f64x2::from(*(cys.add(off) as *const [f64; 2]));
            let y_sum_sq_l = f64x2::from(*(cyss.add(off) as *const [f64; 2]));
            let y_sum_r = f64x2::from(*(ys.add(off) as *const [f64; 2])) - y_sum_l;
            let y_sum_sq_r = f64x2::from(*(yss.add(off) as *const [f64; 2])) - y_sum_sq_l;

            let mean_l = y_sum_l * inv_nl_v;
            let var_l = (y_sum_sq_l * inv_nl_v - mean_l * mean_l).max(zero_v);

            let mean_r = y_sum_r * inv_nr_v;
            let var_r = (y_sum_sq_r * inv_nr_v - mean_r * mean_r).max(zero_v);

            acc += parent_v - wl_v * var_l - wr_v * var_r;
        }
    }

    let mut score = acc.reduce_add();
    for k in (chunks * 2)..n_targets {
        let y_sum_l = cum_y_sums[k];
        let y_sum_sq_l = cum_y_sum_sqs[k];
        let y_sum_r = y_sums_total[k] - y_sum_l;
        let y_sum_sq_r = y_sum_sqs_total[k] - y_sum_sq_l;
        let mean_l = y_sum_l * inv_nl;
        let var_l = f64::max(0.0, y_sum_sq_l * inv_nl - mean_l * mean_l);
        let mean_r = y_sum_r * inv_nr;
        let var_r = f64::max(0.0, y_sum_sq_r * inv_nr - mean_r * mean_r);
        score += parent_vars[k] - wl * var_l - wr * var_r;
    }
    score
}

/// Split score evaluation (256-bit: AVX2)
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn evaluate_split_score_avx2(
    parent_vars: &[f64],
    y_sums_total: &[f64],
    y_sum_sqs_total: &[f64],
    cum_y_sums: &[f64],
    cum_y_sum_sqs: &[f64],
    n_targets: usize,
    inv_nl: f64,
    inv_nr: f64,
    wl: f64,
    wr: f64,
) -> f64 {
    let inv_nl_v = f64x4::splat(inv_nl);
    let inv_nr_v = f64x4::splat(inv_nr);
    let wl_v = f64x4::splat(wl);
    let wr_v = f64x4::splat(wr);
    let zero_v = f64x4::ZERO;

    let chunks = n_targets / 4;
    let mut acc = f64x4::ZERO;

    unsafe {
        let pv = parent_vars.as_ptr();
        let ys = y_sums_total.as_ptr();
        let yss = y_sum_sqs_total.as_ptr();
        let cys = cum_y_sums.as_ptr();
        let cyss = cum_y_sum_sqs.as_ptr();

        for i in 0..chunks {
            let off = i * 4;

            let parent_v = f64x4::from(*(pv.add(off) as *const [f64; 4]));
            let y_sum_l = f64x4::from(*(cys.add(off) as *const [f64; 4]));
            let y_sum_sq_l = f64x4::from(*(cyss.add(off) as *const [f64; 4]));
            let y_sum_r = f64x4::from(*(ys.add(off) as *const [f64; 4])) - y_sum_l;
            let y_sum_sq_r = f64x4::from(*(yss.add(off) as *const [f64; 4])) - y_sum_sq_l;

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
        let var_l = f64::max(0.0, y_sum_sq_l * inv_nl - mean_l * mean_l);
        let mean_r = y_sum_r * inv_nr;
        let var_r = f64::max(0.0, y_sum_sq_r * inv_nr - mean_r * mean_r);
        score += parent_vars[k] - wl * var_l - wr * var_r;
    }
    score
}

/// Split score evaluation (512-bit: AVX-512F)
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn evaluate_split_score_avx512(
    parent_vars: &[f64],
    y_sums_total: &[f64],
    y_sum_sqs_total: &[f64],
    cum_y_sums: &[f64],
    cum_y_sum_sqs: &[f64],
    n_targets: usize,
    inv_nl: f64,
    inv_nr: f64,
    wl: f64,
    wr: f64,
) -> f64 {
    use std::arch::x86_64::*;

    let chunks = n_targets / 8;

    unsafe {
        let inv_nl_v = _mm512_set1_pd(inv_nl);
        let inv_nr_v = _mm512_set1_pd(inv_nr);
        let wl_v = _mm512_set1_pd(wl);
        let wr_v = _mm512_set1_pd(wr);
        let zero_v = _mm512_setzero_pd();
        let mut acc = _mm512_setzero_pd();

        for i in 0..chunks {
            let off = i * 8;

            let parent_v = _mm512_loadu_pd(parent_vars.as_ptr().add(off));
            let y_sum_l = _mm512_loadu_pd(cum_y_sums.as_ptr().add(off));
            let y_sum_sq_l = _mm512_loadu_pd(cum_y_sum_sqs.as_ptr().add(off));

            let y_sum_r = _mm512_sub_pd(_mm512_loadu_pd(y_sums_total.as_ptr().add(off)), y_sum_l);
            let y_sum_sq_r = _mm512_sub_pd(
                _mm512_loadu_pd(y_sum_sqs_total.as_ptr().add(off)),
                y_sum_sq_l,
            );

            // var_l = max(0, sum_sq_l * inv_nl - (sum_l * inv_nl)^2)
            let mean_l = _mm512_mul_pd(y_sum_l, inv_nl_v);
            let var_l = _mm512_max_pd(
                zero_v,
                _mm512_sub_pd(
                    _mm512_mul_pd(y_sum_sq_l, inv_nl_v),
                    _mm512_mul_pd(mean_l, mean_l),
                ),
            );

            let mean_r = _mm512_mul_pd(y_sum_r, inv_nr_v);
            let var_r = _mm512_max_pd(
                zero_v,
                _mm512_sub_pd(
                    _mm512_mul_pd(y_sum_sq_r, inv_nr_v),
                    _mm512_mul_pd(mean_r, mean_r),
                ),
            );

            // acc += parent - wl*var_l - wr*var_r
            let weighted = _mm512_add_pd(_mm512_mul_pd(wl_v, var_l), _mm512_mul_pd(wr_v, var_r));
            acc = _mm512_add_pd(acc, _mm512_sub_pd(parent_v, weighted));
        }

        let mut score = _mm512_reduce_add_pd(acc);

        for k in (chunks * 8)..n_targets {
            let y_sum_l = cum_y_sums[k];
            let y_sum_sq_l = cum_y_sum_sqs[k];
            let y_sum_r = y_sums_total[k] - y_sum_l;
            let y_sum_sq_r = y_sum_sqs_total[k] - y_sum_sq_l;
            let mean_l = y_sum_l * inv_nl;
            let var_l = f64::max(0.0, y_sum_sq_l * inv_nl - mean_l * mean_l);
            let mean_r = y_sum_r * inv_nr;
            let var_r = f64::max(0.0, y_sum_sq_r * inv_nr - mean_r * mean_r);
            score += parent_vars[k] - wl * var_l - wr * var_r;
        }
        score
    }
}

/// Split score evaluation (512-bit fallback)
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn evaluate_split_score_avx512(
    parent_vars: &[f64],
    y_sums_total: &[f64],
    y_sum_sqs_total: &[f64],
    cum_y_sums: &[f64],
    cum_y_sum_sqs: &[f64],
    n_targets: usize,
    inv_nl: f64,
    inv_nr: f64,
    wl: f64,
    wr: f64,
) -> f64 {
    evaluate_split_score_avx2(
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
/// all targets using the widest available SIMD.
///
/// ### Params
///
/// * `parent_vars` - Per-target parent node variance. Length >= n_targets.
/// * `y_sums_total` - Per-target Y sums for the full node.
/// * `y_sum_sqs_total` - Per-target Y squared sums for the full node.
/// * `cum_y_sums` - Cumulative Y sums at the split threshold (already offset
///   to `h_base`). Length >= n_targets.
/// * `cum_y_sum_sqs` - Cumulative Y squared sums at the split threshold.
/// * `n_targets` - Number of active targets.
/// * `inv_nl` - 1.0 / n_left.
/// * `inv_nr` - 1.0 / n_right.
/// * `wl` - n_left / n (left weight).
/// * `wr` - n_right / n (right weight).
///
/// ### Returns
///
/// Sum of per-target weighted variance reductions.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn evaluate_split_score_simd(
    parent_vars: &[f64],
    y_sums_total: &[f64],
    y_sum_sqs_total: &[f64],
    cum_y_sums: &[f64],
    cum_y_sum_sqs: &[f64],
    n_targets: usize,
    inv_nl: f64,
    inv_nr: f64,
    wl: f64,
    wr: f64,
) -> f64 {
    match detect_simd_level() {
        SimdLevel::Avx512 => evaluate_split_score_avx512(
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
        SimdLevel::Avx2 => evaluate_split_score_avx2(
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
        SimdLevel::Sse => evaluate_split_score_sse(
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
        SimdLevel::Scalar => evaluate_split_score_scalar(
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
