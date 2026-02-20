use faer::{Mat, MatRef, RowRef};
use indexmap::IndexSet;
use rayon::prelude::*;
use std::time::Instant;
use wide::{f32x4, f32x8};

use crate::core::math::linear_algebra::linear_regression;
use crate::core::math::stats::{calc_fdr, inv_logit, logit, z_scores_to_pval};
use crate::prelude::*;
use crate::utils::simd::{SimdLevel, detect_simd_level, sum_simd_f32, sum_squares_simd_f32};

/////////////
// Hotspot //
/////////////

//////////
// SIMD //
//////////

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
/// Used in compute_local_cov_max: sum(a[i] * b[i] * b[i])
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
// Params //
////////////

/// HotSpot parameters
///
/// ### Fields
///
/// * `model` - The model to use for modelling the GEX. Choice of
///   `"danb"`, `"bernoulli"` or `"normal"`.
/// * `normalise` - Shall the data be normalised.
/// * `knn_params` - The knnParams via the `KnnParams` structure.
pub struct HotSpotParams {
    // hotspot parameters
    pub model: String,
    pub normalise: bool,
    // knn params
    pub knn_params: KnnParams,
}

/////////////
// Helpers //
/////////////

#[derive(Debug, Clone)]
pub enum GexModel {
    /// Use depth-adjusted negative binomial model
    DephAdjustNegBinom,
    /// Uses Bernoulli distribution to model prediction probability
    Bernoulli,
    /// Use depth-adjusted normal model
    Normal,
}

/// Parse the model to use gene expression
///
/// ### Params
///
/// * `s` - Type of model to use the model
///
/// ### Returns
///
/// Option of the GexModel to use (some not yet implemented)
pub fn parse_gex_model(s: &str) -> Option<GexModel> {
    match s.to_lowercase().as_str() {
        "danb" => Some(GexModel::DephAdjustNegBinom),
        "bernoulli" => Some(GexModel::Bernoulli),
        "normal" => Some(GexModel::Normal),
        _ => None,
    }
}

/// Structure for the gene results
///
/// ### Fields
///
/// * `gene_idx` - Gene index of the analysed gene
/// * `c` - Geary's C statistic for this gene
/// * `z` - Z-score for this gene
/// * `pval` - P-value based on the Z-score
/// * `fdr` - False discovery corrected pvals
#[derive(Debug, Clone)]
pub struct HotSpotGeneRes {
    pub gene_idx: Vec<usize>,
    pub c: Vec<f64>,
    pub z: Vec<f64>,
    pub pval: Vec<f64>,
    pub fdr: Vec<f64>,
}

/// Structure for pair-wise correlations
///
/// ### Fields
///
/// * `cor` - Symmetric matrix with cor coefficients (N_genes x N_genes)
/// * `z_scores` - Symmetric matrix with Z scores (N_genex x N_genes)
#[derive(Debug, Clone)]
pub struct HotSpotPairRes {
    pub cor: Mat<f32>,
    pub z_scores: Mat<f32>,
}

/// Compute momentum weights
///
/// Calculates the expected value (EG) and expected squared value (EG2) of the
/// local covariance statistic under the null hypothesis of no spatial
/// autocorrelation.
///
/// ### Params
///
/// * `mu` - Mean expression values for each cell
/// * `x2` - Second moment (variance + mean²) for each cell
/// * `neighbours` - Neighbour indices for each cell
/// * `weights` - Edge weights for each neighbour connection
///
/// ### Returns
///
/// Tuple of (EG, EG2) where:
///
/// - EG: Expected value of the spatial covariance statistic
/// - EG2: Expected value of the squared spatial covariance statistic
fn compute_moments_weights(
    mu: &[f32],
    x2: &[f32],
    neighbours: &[Vec<usize>],
    weights: &[Vec<f32>],
) -> (f32, f32) {
    let n = neighbours.len();
    let mu_sq: Vec<f32> = mu.iter().map(|&m| m * m).collect();

    let mut eg = 0_f32;
    let mut t1 = vec![0_f32; n];
    let mut t2 = vec![0_f32; n];

    for i in 0..n {
        let mu_i = mu[i];

        for (k, &j) in neighbours[i].iter().enumerate() {
            let wij = weights[i][k];
            let mu_j = mu[j];

            eg += wij * mu_i * mu_j;

            t1[i] += wij * mu_j;
            let wij_sq = wij * wij;
            t2[i] += wij_sq * mu_j * mu_j;

            // Add these back:
            t1[j] += wij * mu_i;
            t2[j] += wij_sq * mu_i * mu_i;
        }
    }

    let mut eg2 = 0_f32;

    for i in 0..n {
        eg2 += (x2[i] - mu_sq[i]) * (t1[i] * t1[i] - t2[i]);
    }

    for i in 0..n {
        let x2_i = x2[i];
        let mu_sq_i = mu_sq[i];

        for (k, &j) in neighbours[i].iter().enumerate() {
            let wij = weights[i][k];
            eg2 += wij * wij * (x2_i * x2[j] - mu_sq_i * mu_sq[j]);
        }
    }

    eg2 += eg * eg;

    (eg, eg2)
}

/// Remove redundancy in bidirectional edge weights
///
/// Consolidates weights from bidirectional edges by accumulating both
/// directions into the lower-indexed node's edge and zeroing the
/// higher-indexed node's reciprocal edge.
///
/// ### Params
///
/// * `neighbours` - Neighbour indices for each node
/// * `weights` - Edge weights for each neighbour connection
///
/// ### Returns
///
/// Modified weights with redundant edges zeroed
fn make_weights_non_redundant(neighbours: &[Vec<usize>], weights: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut w_no_redundant = weights.to_vec();

    for i in 0..neighbours.len() {
        for k in 0..neighbours[i].len() {
            let j = neighbours[i][k];

            if j < i {
                continue;
            }

            // check if j has i as a neighbour
            for k2 in 0..neighbours[j].len() {
                if neighbours[j][k2] == i {
                    let w_ji = w_no_redundant[j][k2];
                    w_no_redundant[j][k2] = 0.0;
                    w_no_redundant[i][k] += w_ji;
                    break;
                }
            }
        }
    }

    w_no_redundant
}

/// Compute node degree from edge weights
///
/// Calculates the degree (sum of incident edge weights) for each node.
/// Each edge contributes to the degree of both its endpoints.
///
/// ### Params
///
/// * `neighbours` - Neighbour indices for each node
/// * `weights` - Edge weights for each neighbour connection
///
/// ### Returns
///
/// Vector of degree values for each node
fn compute_node_degree(neighbours: &[Vec<usize>], weights: &[Vec<f32>]) -> Vec<f32> {
    let mut d = vec![0.0_f32; neighbours.len()];

    for i in 0..neighbours.len() {
        for k in 0..neighbours[i].len() {
            let j = neighbours[i][k];
            let w_ij = weights[i][k];

            d[i] += w_ij;
            d[j] += w_ij;
        }
    }

    d
}

/// Compute local covariance using edge weights
///
/// Calculates the weighted local covariance statistic for spatial
/// autocorrelation. This is the numerator of Geary's C statistic.
///
/// ### Params
///
/// * `vals` - Gene expression values for each cell
/// * `neighbours` - Neighbour indices for each cell
/// * `weights` - Edge weights for each neighbour connection
///
/// ### Returns
///
/// The local covariance statistic
fn local_cov_weights(vals: &[f32], neighbours: &[Vec<usize>], weights: &[Vec<f32>]) -> f32 {
    let mut out = 0.0;

    for i in 0..vals.len() {
        let xi = vals[i];
        if xi == 0.0 {
            continue;
        }

        for (k, &j) in neighbours[i].iter().enumerate() {
            let xj = vals[j];
            let wij = weights[i][k];

            if xj != 0.0 && wij != 0.0 {
                out += xi * xj * wij;
            }
        }
    }

    out
}

/// Compute maximum possible local covariance
///
/// Calculates the theoretical maximum value of the local covariance statistic
/// given the node degrees and expression values. Used to normalise Geary's C.
///
/// ### Params
///
/// * `node_degrees` - Sum of edge weights for each node
/// * `vals` - Gene expression values for each cell
///
/// ### Returns
///
/// Maximum possible local covariance
fn compute_local_cov_max(node_degrees: &[f32], vals: &[f32]) -> f32 {
    fused_mul_square_sum_simd(node_degrees, vals) / 2.0
}

/// Center (Z-score) the values
///
/// Transforms values to have zero means and unit variance of one using the
/// provided stats.
///
/// ### Params
///
/// * `vals` - Mutable reference to the values to scale
/// * `mu` - The mean values
/// * `var` - The variance of the values
fn center_values(vals: &mut [f32], mu: &[f32], var: &[f32]) {
    assert_same_len!(vals, mu, var);

    center_values_simd(vals, mu, var);
}

//////////////////
// Corr helpers //
//////////////////

/// Compute local covariance for gene pairs
///
/// Test statistic for local pairwise autocorrelation. Calculates the weighted
/// covariance between two genes across neighbouring cells.
///
/// ### Params
///
/// * `x` - RowRef for first gene.
/// * `y` - RowRef for second gene.
/// * `neighbours` - Neighbour indices for each cell
/// * `weights` - Edge weights for each neighbour connection
///
/// ### Returns
///
/// Local covariance statistic
fn local_cov_pair(
    x: RowRef<f32>,
    y: RowRef<f32>,
    neighbours: &[Vec<usize>],
    weights: &[Vec<f32>],
) -> f32 {
    let mut out = 0.0;

    for i in 0..x.ncols() {
        let xi = x[i];
        let yi = y[i];
        if xi == 0.0 && yi == 0.0 {
            continue;
        }
        for k in 0..neighbours[i].len() {
            let j = neighbours[i][k];
            let w_ij = weights[i][k];

            let xj = x[j];
            let yj = y[j];

            out += w_ij * (xi * yj + yi * xj) / 2.0;
        }
    }

    out
}

/// Compute local covariance for gene pairs
///
/// Test statistic for local pairwise autocorrelation. Calculates the weighted
/// covariance between two genes across neighbouring cells.
///
/// ### Params
///
/// * `x` - Slice for first gene.
/// * `y` - Slice for second gene.
/// * `neighbours` - Neighbour indices for each cell
/// * `weights` - Edge weights for each neighbour connection
///
/// ### Returns
///
/// Local covariance statistic
fn local_cov_pair_vec(
    x: &[f32],
    y: &[f32],
    neighbours: &[Vec<usize>],
    weights: &[Vec<f32>],
) -> f32 {
    neighbours
        .iter()
        .zip(weights.iter())
        .enumerate()
        .map(|(i, (neighs, ws))| {
            let xi = x[i];
            let yi = y[i];
            neighs
                .iter()
                .zip(ws.iter())
                .map(|(&j, &w)| {
                    let xj = x[j];
                    let yj = y[j];
                    w * (xi * yj + yi * xj) / 2.0
                })
                .sum::<f32>()
        })
        .sum()
}

/// Compute conditional EG2 for correlation
///
/// Calculates the expected value of G_square for the conditional correlation
/// statistic, assuming standardised variables.
///
/// ### Params
///
/// * `x` - Standardised expression values for a gene
/// * `neighbors` - Neighbour indices for each cell
/// * `weights` - Edge weights for each neighbour connection
///
/// ### Returns
///
/// Expected value of G²
fn conditional_eg2(x: &[f32], neighbours: &[Vec<usize>], weights: &[Vec<f32>]) -> f32 {
    let n = neighbours.len();

    let mut t1x = vec![0_f32; n];

    for i in 0..n {
        for k in 0..neighbours[i].len() {
            let j = neighbours[i][k];
            let wij = weights[i][k];

            if wij == 0.0 {
                continue;
            }

            t1x[i] += wij * x[j];
            t1x[j] += wij * x[i];
        }
    }

    t1x.iter().map(|&t| t * t).sum()
}

/// Compute maximum possible pairwise local covariance
///
/// Calculates the theoretical maximum for pairwise correlation normalisation.
///
/// ### Params
///
/// * `node_degrees` - Sum of edge weights for each node
/// * `counts` - Centred gene expression matrix (genes × cells)
///
/// ### Returns
///
/// Matrix of maximum covariances (genes × genes)
fn compute_local_cov_pairs_max(node_degrees: &[f32], counts: &Mat<f32>) -> Mat<f32> {
    let n_genes = counts.nrows();

    let gene_maxs: Vec<f32> = (0..n_genes)
        .into_par_iter()
        .map(|i| {
            let row = counts.row(i);
            let row_vec = row.iter().copied().collect::<Vec<f32>>();
            compute_local_cov_max(node_degrees, &row_vec)
        })
        .collect();

    let values: Vec<f32> = (0..n_genes * n_genes)
        .into_par_iter()
        .map(|idx| {
            let i = idx / n_genes;
            let j = idx % n_genes;
            (gene_maxs[i] + gene_maxs[j]) / 2.0
        })
        .collect();

    let mut result = Mat::zeros(n_genes, n_genes);
    for (idx, &val) in values.iter().enumerate() {
        result[(idx / n_genes, idx % n_genes)] = val;
    }

    result
}

/// Centre gene counts for correlation computation
///
/// Standardises gene expression using the specified model, transforming to
/// zero mean and unit variance.
///
/// ### Params
///
/// * `gene` - Reference to gene expression data
/// * `umi_counts` - Total UMI counts per cell
/// * `n_cells` - Number of cells
/// * `model` - Statistical model to use
///
/// ### Returns
///
/// Vector of centred expression values
fn create_centered_counts_gene(
    gene: &CscGeneChunk,
    umi_counts: &[f32],
    n_cells: usize,
    model: &GexModel,
) -> Vec<f32> {
    let (mu, var, _) = match model {
        GexModel::DephAdjustNegBinom => danb_model(gene, umi_counts, n_cells),
        GexModel::Bernoulli => bernoulli_model(gene, umi_counts, n_cells),
        GexModel::Normal => normal_model(gene, umi_counts, n_cells),
    };

    let mut vals = vec![0_f32; n_cells];
    for (&idx, &val) in gene.indices.iter().zip(&gene.data_raw) {
        vals[idx as usize] = val as f32;
    }

    center_values(&mut vals, &mu, &var);

    vals
}

////////////////
// DANB model //
////////////////

/// Depth-adjusted negative binomial (DANB) model
///
/// Fits a negative binomial distribution to gene expression data, adjusting
/// for sequencing depth differences between cells.
///
/// ### Params
///
/// * `gene` - Reference to the CscGeneChunk on which to apply the model.
/// * `umi_counts` - Slice of the UMI counts across these cells (i.e.,
///   sequencing depth).
/// * `n_cells` - Total number of cells
///
/// ### Returns
///
/// Tuple of (mu, var, x2) where:
/// - mu: Mean expression for each cell
/// - var: Variance for each cell
/// - x2: Second moment (var + mu²) for each cell
fn danb_model(
    gene: &CscGeneChunk,
    umi_counts: &[f32],
    n_cells: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = n_cells as f32;
    let total: f32 = sum_simd_f32(umi_counts);
    let tj: f32 = gene.data_raw.iter().map(|&x| x as f32).sum();

    let mu: Vec<f32> = umi_counts.iter().map(|&ti| tj * ti / total).collect();

    // Build dense array for O(1) lookups
    let mut data_dense = vec![0.0f32; n_cells];
    for (&idx, &val) in gene.indices.iter().zip(&gene.data_raw) {
        data_dense[idx as usize] = val as f32;
    }

    let mut sum_sq = 0_f32;
    for i in 0..n_cells {
        let diff = data_dense[i] - mu[i];
        sum_sq += diff * diff;
    }

    let vv = sum_sq / (n - 1.0);
    let tis_sq_sum: f32 = sum_squares_simd_f32(umi_counts);
    let mut size = ((tj * tj) / total) * (tis_sq_sum / total) / ((n - 1.0) * vv - tj);

    if size < 0.0 {
        size = 1e9;
    } else if size < 1e-10 {
        size = 1e-10;
    }

    let var: Vec<f32> = mu.iter().map(|&m| m * (1.0 + m / size)).collect();
    let x2: Vec<f32> = var.iter().zip(&mu).map(|(&v, &m)| v + m * m).collect();

    (mu, var, x2)
}

/////////////////////
// Bernoulli model //
/////////////////////

/// Bin gene detections by UMI count bins
///
/// Calculates the detection rate within each bin, applying Laplace smoothing
/// to handle edge cases (0% or 100% detection).
///
/// ### Params
///
/// * `detected_gene` - Binary detection indicators (0 or 1) for each cell
/// * `umi_count_bins` - Bin assignment for each cell
/// * `n_bins` - Total number of bins
///
/// ### Returns
///
/// Vector of detection rates per bin (with Laplace smoothing)
fn bin_gene_detection(detected_gene: &[f32], umi_count_bins: &[usize], n_bins: usize) -> Vec<f32> {
    let mut bin_detects = vec![0_f32; n_bins];
    let mut bin_totals = vec![0_f32; n_bins];

    for i in 0..detected_gene.len() {
        let bin_i = umi_count_bins[i];
        bin_detects[bin_i] += detected_gene[i];
        bin_totals[bin_i] += 1.0;
    }

    // laplace smoothing
    bin_detects
        .iter()
        .zip(&bin_totals)
        .map(|(&d, &t)| (d + 1.0) / (t + 2.0))
        .collect()
}

/// Quantile-based binning with duplicate edge handling
///
/// Generates quantile-based bins from data, dropping duplicate bin edges when
/// they would result in empty bins.
///
/// ### Params
///
/// * `data` - Input data to bin
/// * `n_bins` - Target number of bins
///
/// ### Returns
///
/// Tuple of (bin_assignments, bin_edges) where:
/// - bin_assignments: Vector of bin indices for each data point
/// - bin_edges: Vector of bin edge values (length = n_bins + 1)
fn quantile_cut(data: &[f32], n_bins: usize) -> (Vec<usize>, Vec<f32>) {
    let mut data_sorted = data.to_vec();
    data_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = data_sorted.len();
    let mut edges = vec![data_sorted[0]];

    for i in 1..n_bins {
        let idx = (i * n) / n_bins;
        let value = data_sorted[idx.min(n - 1)];

        if value > *edges.last().unwrap() {
            edges.push(value);
        }
    }

    let max_val = data_sorted[n - 1];
    if max_val > *edges.last().unwrap() {
        edges.push(max_val + 1e-6);
    } else {
        *edges.last_mut().unwrap() += 1e-6;
    }

    let n_actual_bins = edges.len() - 1;

    // binary search is faster here...
    let bin_assignments: Vec<usize> = data
        .iter()
        .map(|&x| {
            edges
                .partition_point(|&edge| edge <= x)
                .saturating_sub(1)
                .min(n_actual_bins - 1)
        })
        .collect();

    (bin_assignments, edges)
}

/// Bernoulli model for gene expression
///
/// Models the probability of detecting gene expression using a Bernoulli
/// distribution. Fits a logistic regression model on binned UMI counts to
/// predict detection probability.
///
/// ### Params
///
/// * `gene` - Reference to the CscGeneChunk containing gene expression data
/// * `umi_counts` - Total UMI counts per cell
/// * `n_cells` - Total number of cells
///
/// ### Returns
///
/// Tuple of (mu, var, x2) where:
/// - mu: Detection probability for each cell
/// - var: Variance (p * (1-p)) for each cell
/// - x2: Second moment (equal to mu for Bernoulli)
fn bernoulli_model(
    gene: &CscGeneChunk,
    umi_counts: &[f32],
    n_cells: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    const N_BIN_TARGET: usize = 30;

    let mut detected_gene = vec![0_f32; n_cells];
    for idx in &gene.indices {
        detected_gene[*idx as usize] = 1.0;
    }

    let log_umi: Vec<f32> = umi_counts
        .iter()
        .map(|&x| if x > 0.0 { x.log10() } else { 0.0 })
        .collect();

    let (umi_count_bins, bin_edges) = quantile_cut(&log_umi, N_BIN_TARGET);
    let n_bins = bin_edges.len() - 1;

    let bin_centers: Vec<f32> = (0..n_bins)
        .map(|i| (bin_edges[i] + bin_edges[i + 1]) / 2.0)
        .collect();

    let bin_detects = bin_gene_detection(&detected_gene, &umi_count_bins, n_bins);

    let lbin_detects: Vec<f32> = bin_detects.iter().map(|&p| logit(p)).collect();
    let coef = linear_regression(&bin_centers, &lbin_detects);

    let mu: Vec<f32> = log_umi
        .iter()
        .map(|&log_u| inv_logit(coef.0 + coef.1 * log_u))
        .collect();

    let var: Vec<f32> = mu.iter().map(|&p| p * (1.0 - p)).collect();
    let x2: Vec<f32> = mu.clone();

    (mu, var, x2)
}

//////////////////
// Normal model //
//////////////////

/// Normal model for gene expression
///
/// Simplest model just using the normalised counts in the data.
///
/// ### Params
///
/// * `gene` - Reference to the CscGeneChunk containing gene expression data
/// * `n_cells` - Total number of cells
///
/// ### Returns
///
/// Tuple of (mu, var, x2) where:
/// - mu: Mean expression for each cell (from linear regression)
/// - var: Residual variance (constant across cells)
/// - x2: Second moment (var + mu²) for each cell
fn normal_model(
    gene: &CscGeneChunk,
    umi_counts: &[f32],
    n_cells: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut gene_raw = vec![0_f32; n_cells];
    for (&idx, &val) in gene.indices.iter().zip(&gene.data_raw) {
        gene_raw[idx as usize] = val as f32;
    }

    // Fit linear regression: expression ~ log(umi_counts)
    let log_umi: Vec<f32> = umi_counts
        .iter()
        .map(|&x| if x > 0.0 { x.ln() } else { 0.0 })
        .collect();

    let (intercept, slope) = linear_regression(&log_umi, &gene_raw);

    // Cell-specific mu from regression
    let mu: Vec<f32> = log_umi.iter().map(|&x| intercept + slope * x).collect();

    // Residual variance (constant across cells)
    let residuals_sq: f32 = gene_raw
        .iter()
        .zip(&mu)
        .map(|(&obs, &pred)| (obs - pred).powi(2))
        .sum();
    let var_val = residuals_sq / (n_cells as f32 - 2.0);

    let var = vec![var_val; n_cells];
    let x2: Vec<f32> = mu.iter().map(|&m| var_val + m * m).collect();

    (mu, var, x2)
}

//////////
// Main //
//////////

/// HotSpot structure
///
/// Main structure for computing spatial autocorrelation and gene <> gene
/// correlations in spatially-resolved transcriptomics data.
///
/// ### Fields
///
/// * `f_path_gene` - File path to the gene-based binary file.
/// * `f_path_cell` - File path to the cell-based binary file.
/// * `neigbours` - Slice if the indices of the cells to include in this
///   analysis.
/// * `weights` - Slice of the distances to the neighbours of a given cell.
/// * `cells_to_keep` - Slice of cells to analyse/keep in this analysis.
/// * `node_degrees` - Pre-computed node-degree for each cell based on the
///   weights.
/// * `umi_counts` - Optional vector with the total UMI counts per cell
/// * `wtot2` -
/// * `n_cells` - Total number of cells analysed in the experiment.
#[derive(Clone, Debug)]
pub struct Hotspot<'a> {
    f_path_gene: String,
    f_path_cell: String,
    neighbours: &'a [Vec<usize>],
    weights: Vec<Vec<f32>>,
    cells_to_keep: &'a [usize],
    node_degrees: Vec<f32>,
    umi_counts: Option<Vec<f32>>,
    wtot2: f32,
    n_cells: usize,
}

impl<'a> Hotspot<'a> {
    /// Initialise a new instance
    ///
    /// ### Params
    ///
    /// * `f_path_gene` - File path to the gene-based binary file.
    /// * `f_path_cell` - File path to the cell-based binary file.
    /// * `cells_to_keep` - Slice if the indices of the cells to include in this
    ///   analysis.
    /// * `neighbours` - Slice of the indices of the neighbours of the given
    ///   cell.
    /// * `weights` - Slice of the distances to the neighbours of a given cell.
    ///
    /// ### Return
    ///
    /// Initialised `HotSpot` class.
    pub fn new(
        f_path_gene: String,
        f_path_cell: String,
        cells_to_keep: &'a [usize],
        neighbours: &'a [Vec<usize>],
        weights: &mut [Vec<f32>],
    ) -> Self {
        let n_cells = neighbours.len();

        let weights = make_weights_non_redundant(neighbours, weights);

        let node_degrees = compute_node_degree(neighbours, &weights);

        let wtot2: f32 = weights.iter().flatten().map(|&w| w * w).sum();

        Self {
            f_path_gene,
            f_path_cell,
            neighbours,
            weights,
            cells_to_keep,
            node_degrees,
            umi_counts: None,
            wtot2,
            n_cells,
        }
    }

    /// Compute spatial autocorrelation for all specified genes
    ///
    /// Calculates Geary's C statistic and Z-scores for spatial autocorrelation
    /// across the specified genes.
    ///
    /// ### Params
    ///
    /// * `gene_indices` - Indices of genes to analyse
    /// * `model` - Statistical model to use ("danb", "bernoulli", or "normal")
    /// * `centered` - Whether to centre the data before computing statistics
    /// * `verbose` - Whether to print progress information
    ///
    /// ### Returns
    ///
    /// `Result<HotSpotGeneRes>` with gene indices, Geary's C, Z-scores, derived
    /// p-values and FDR.
    pub fn compute_all_genes(
        &mut self,
        gene_indices: &[usize],
        model: &str,
        centered: bool,
        verbose: bool,
    ) -> Result<HotSpotGeneRes, String> {
        let gex_model =
            parse_gex_model(model).ok_or_else(|| format!("Invalid model type: {}", model))?;

        self.populate_umi_counts();

        let cell_set: IndexSet<u32> = self.cells_to_keep.iter().map(|&x| x as u32).collect();

        let start_reading = Instant::now();

        let reader = ParallelSparseReader::new(&self.f_path_gene).unwrap();
        let mut gene_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(gene_indices);

        gene_chunks.par_iter_mut().for_each(|chunk| {
            chunk.filter_selected_cells(&cell_set);
        });

        let end_reading = start_reading.elapsed();

        if verbose {
            println!("Loaded in data: {:.2?}", end_reading);
        }

        let start_calculation = Instant::now();

        let res: Vec<(usize, f32, f32)> = gene_chunks
            .par_iter()
            .map(|chunk| self.compute_single_gene(chunk, &gex_model, centered))
            .collect();

        let mut gene_indices: Vec<usize> = Vec::with_capacity(res.len());
        let mut gaery_c: Vec<f64> = Vec::with_capacity(res.len());
        let mut z_scores: Vec<f64> = Vec::with_capacity(res.len());

        for (idx, c, z) in res {
            if !z.is_nan() {
                gene_indices.push(idx);
                gaery_c.push(c as f64);
                z_scores.push(z as f64);
            }
        }

        let end_calculations = start_calculation.elapsed();

        if verbose {
            println!("Finsished the calculations: {:.2?}", end_calculations);
        }

        let p_vals = z_scores_to_pval(&z_scores, "twosided");
        let fdrs = calc_fdr(&p_vals);

        Ok(HotSpotGeneRes {
            gene_idx: gene_indices,
            c: gaery_c,
            z: z_scores,
            pval: p_vals,
            fdr: fdrs,
        })
    }

    /// Compute spatial autocorrelation with streaming (memory-efficient)
    ///
    /// Processes genes in batches to reduce memory usage for large datasets.
    ///
    /// ### Params
    ///
    /// * `gene_indices` - Indices of genes to analyse
    /// * `model` - Statistical model to use ("danb", "bernoulli", or "normal")
    /// * `centered` - Whether to centre the data before computing statistics
    /// * `verbose` - Whether to print progress information
    ///
    /// ### Returns
    ///
    /// `Result<HotSpotGeneRes>` with gene indices, Geary's C, Z-scores, derived
    /// p-values and FDR.
    pub fn compute_all_genes_streaming(
        &mut self,
        gene_indices: &[usize],
        model: &str,
        centered: bool,
        verbose: bool,
    ) -> Result<HotSpotGeneRes, String> {
        const GENE_BATCH_SIZE: usize = 1000;

        let start_all = Instant::now();

        let no_genes = gene_indices.len();
        let no_batches = no_genes.div_ceil(GENE_BATCH_SIZE);
        let cell_set: IndexSet<u32> = self.cells_to_keep.iter().map(|&x| x as u32).collect();
        let reader = ParallelSparseReader::new(&self.f_path_gene).unwrap();

        let gex_model =
            parse_gex_model(model).ok_or_else(|| format!("Invalid model type: {}", model))?;

        self.populate_umi_counts();

        let mut results: Vec<(Vec<usize>, Vec<f64>, Vec<f64>)> = Vec::with_capacity(no_batches);

        for batch_idx in 0..no_batches {
            if verbose && batch_idx % 5 == 0 {
                let progress = (batch_idx + 1) as f32 / no_batches as f32 * 100.0;
                println!("  Progress: {:.1}%", progress);
            }

            let start_gene = batch_idx * GENE_BATCH_SIZE;
            let end_gene = ((batch_idx + 1) * GENE_BATCH_SIZE).min(no_genes);

            let gene_indices: Vec<usize> = (start_gene..end_gene).collect();

            let start_loading = Instant::now();

            let mut gene_chunks = reader.read_gene_parallel(&gene_indices);

            gene_chunks.par_iter_mut().for_each(|chunk| {
                chunk.filter_selected_cells(&cell_set);
            });

            let end_loading = start_loading.elapsed();

            if verbose {
                println!("   Loaded batch in: {:.2?}.", end_loading);
            }

            let start_calc = Instant::now();

            let batch_res: Vec<(usize, f32, f32)> = gene_chunks
                .par_iter()
                .map(|chunk| self.compute_single_gene(chunk, &gex_model, centered))
                .collect();

            let mut batch_gene_indices: Vec<usize> = Vec::with_capacity(batch_res.len());
            let mut batch_gaery_c: Vec<f64> = Vec::with_capacity(batch_res.len());
            let mut batch_z_scores: Vec<f64> = Vec::with_capacity(batch_res.len());

            for (idx, c, z) in batch_res {
                if z.is_finite() {
                    batch_gene_indices.push(idx);
                    batch_gaery_c.push(c as f64);
                    batch_z_scores.push(z as f64);
                }
            }

            let end_calc = start_calc.elapsed();

            if verbose {
                println!("   Finished calculations in: {:.2?}.", end_calc);
            }

            results.push((batch_gene_indices, batch_gaery_c, batch_z_scores));
        }

        let mut gene_indices: Vec<usize> = Vec::new();
        let mut gaery_c: Vec<f64> = Vec::new();
        let mut z_scores: Vec<f64> = Vec::new();

        for (idx, c, z) in results {
            gene_indices.extend(idx);
            gaery_c.extend(c);
            z_scores.extend(z);
        }

        let p_vals = z_scores_to_pval(&z_scores, "twosided");
        let fdrs = calc_fdr(&p_vals);

        let end_total = start_all.elapsed();

        if verbose {
            println!("Finished the full run in : {:.2?}.", end_total);
        }

        Ok(HotSpotGeneRes {
            gene_idx: gene_indices,
            c: gaery_c,
            z: z_scores,
            pval: p_vals,
            fdr: fdrs,
        })
    }

    /// Compute a single gene's spatial autocorrelation
    ///
    /// Internal method for calculating Geary's C and Z-score for one gene.
    ///
    /// ### Params
    ///
    /// * `gene_chunk` - Gene expression data
    /// * `gex_model` - Statistical model to apply
    /// * `centered` - Whether to centre the data
    ///
    /// ### Returns
    ///
    /// Tuple of (gene_index, Geary's C, Z-score)
    fn compute_single_gene(
        &self,
        gene_chunk: &CscGeneChunk,
        gex_model: &GexModel,
        centered: bool,
    ) -> (usize, f32, f32) {
        assert!(
            self.umi_counts.is_some(),
            "The internal UMI counts need to be populated"
        );

        let (mu, var, x2) = match gex_model {
            GexModel::DephAdjustNegBinom => {
                danb_model(gene_chunk, self.umi_counts.as_ref().unwrap(), self.n_cells)
            }
            GexModel::Bernoulli => {
                bernoulli_model(gene_chunk, self.umi_counts.as_ref().unwrap(), self.n_cells)
            }
            GexModel::Normal => {
                normal_model(gene_chunk, self.umi_counts.as_ref().unwrap(), self.n_cells)
            }
        };

        let mut vals = vec![0_f32; self.n_cells];
        for (&idx, &val) in gene_chunk.indices.iter().zip(&gene_chunk.data_raw) {
            vals[idx as usize] = val as f32;
        }

        if centered {
            center_values(&mut vals, &mu, &var);
        }

        let g = local_cov_weights(&vals, self.neighbours, &self.weights);

        let (eg, eg2) = if centered {
            (0.0, self.wtot2)
        } else {
            compute_moments_weights(&mu, &x2, self.neighbours, &self.weights)
        };

        let std_g = (eg2 - eg * eg).sqrt();
        let z = (g - eg) / std_g;

        let g_max = compute_local_cov_max(&self.node_degrees, &vals);
        let c = (g - eg) / g_max;

        (gene_chunk.original_index, c, z)
    }

    /// Compute pairwise gene correlations (in-memory version)
    ///
    /// Calculates local spatial correlations between all pairs of specified
    /// genes. Loads all gene data into memory for faster computation.
    /// WARNING: This will create a dense matrix of size n_cells x n_genes
    /// in memory! Should only be used for small data sets or very selected
    /// number of genes!
    ///
    /// ### Params
    ///
    /// * `gene_indices` - Indices of genes to analyse
    /// * `model` - Statistical model to use ("danb", "bernoulli", or "normal")
    /// * `verbose` - Whether to print progress information
    ///
    /// ### Returns
    ///
    /// Result containing HotSpotPairRes with correlation and Z-score matrices
    pub fn compute_gene_cor(
        &mut self,
        gene_indices: &[usize],
        model: &str,
        verbose: bool,
    ) -> Result<HotSpotPairRes, String> {
        let gex_model =
            parse_gex_model(model).ok_or_else(|| format!("Invalid model type: {}", model))?;

        self.populate_umi_counts();

        let cell_set: IndexSet<u32> = self.cells_to_keep.iter().map(|&x| x as u32).collect();

        if verbose {
            println!("Loading {} genes...", gene_indices.len());
        }

        let start_loading = Instant::now();
        let reader = ParallelSparseReader::new(&self.f_path_gene).unwrap();
        let mut gene_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(gene_indices);

        gene_chunks.par_iter_mut().for_each(|chunk| {
            chunk.filter_selected_cells(&cell_set);
        });

        if verbose {
            println!("Loaded data in {:.2?}", start_loading.elapsed());
            println!("Centering gene expression...");
        }

        let start_center = Instant::now();
        let centered_counts: Vec<Vec<f32>> = gene_chunks
            .par_iter()
            .map(|gene| {
                create_centered_counts_gene(
                    gene,
                    self.umi_counts.as_ref().unwrap(),
                    self.n_cells,
                    &gex_model,
                )
            })
            .collect();

        let n_genes = centered_counts.len();
        let mut counts_mat = Mat::zeros(n_genes, self.n_cells);
        for (i, gene_vec) in centered_counts.iter().enumerate() {
            for (j, &val) in gene_vec.iter().enumerate() {
                counts_mat[(i, j)] = val;
            }
        }

        if verbose {
            println!("Centered in {:.2?}", start_center.elapsed());
            println!("Computing conditional EG2 values...");
        }

        let start_eg2 = Instant::now();
        let eg2s: Vec<f32> = (0..n_genes)
            .into_par_iter()
            .map(|i| {
                let row = counts_mat.row(i);
                let row_vec = row.iter().copied().collect::<Vec<f32>>();
                conditional_eg2(&row_vec, self.neighbours, &self.weights)
            })
            .collect();

        if verbose {
            println!("Computed EG2 in {:.2?}", start_eg2.elapsed());
            println!("Computing pairwise correlations...");
        }

        let start_pairs = Instant::now();
        let n_pairs = (n_genes * (n_genes - 1)) / 2;

        let pairs: Vec<(usize, usize)> = (0..n_genes)
            .flat_map(|i| ((i + 1)..n_genes).map(move |j| (i, j)))
            .collect();

        let results: Vec<(usize, usize, f32, f32)> = pairs
            .par_iter()
            .map(|&(i, j)| {
                let x = counts_mat.row(i);
                let y = counts_mat.row(j);

                let lc = local_cov_pair(x, y, self.neighbours, &self.weights) * 2.0;

                // Use the minimum of the two Z-scores (more conservative)
                let eg = 0.0;

                let stdg_xy = eg2s[i].sqrt();
                let z_xy = (lc - eg) / stdg_xy;

                let stdg_yx = eg2s[j].sqrt();
                let z_yx = (lc - eg) / stdg_yx;

                let z = if z_xy.abs() < z_yx.abs() { z_xy } else { z_yx };

                (i, j, lc, z)
            })
            .collect();

        if verbose {
            println!(
                "Computed {} pairs in {:.2?}",
                n_pairs,
                start_pairs.elapsed()
            );
            println!("Building matrices...");
        }

        // generate symmetric matrices
        let mut lc_mat = Mat::zeros(n_genes, n_genes);
        let mut z_mat = Mat::zeros(n_genes, n_genes);

        for (i, j, lc, z) in results {
            if z.is_finite() {
                lc_mat[(i, j)] = lc;
                lc_mat[(j, i)] = lc;
                z_mat[(i, j)] = z;
                z_mat[(j, i)] = z;
            }
        }

        let lc_maxs = compute_local_cov_pairs_max(&self.node_degrees, &counts_mat);
        for i in 0..n_genes {
            for j in 0..n_genes {
                lc_mat[(i, j)] /= lc_maxs[(i, j)];
            }
        }

        if verbose {
            println!("Done!");
        }

        Ok(HotSpotPairRes {
            cor: lc_mat,
            z_scores: z_mat,
        })
    }

    /// Compute pairwise gene correlations (streaming version)
    ///
    /// Calculates local spatial correlations between all pairs of specified
    /// genes. Loads all gene data into memory for faster computation. Due to
    /// the nature of the problem, the function will calculate the correlation
    /// matrices in two passes with heavy nesting.
    ///
    /// ### Params
    ///
    /// * `gene_indices` - Indices of genes to analyse
    /// * `model` - Statistical model to use ("danb", "bernoulli", or "normal")
    /// * `verbose` - Whether to print progress information
    ///
    /// ### Returns
    ///
    /// Result containing HotSpotPairRes with correlation and Z-score matrices
    pub fn compute_gene_cor_streaming(
        &mut self,
        gene_indices: &[usize],
        model: &str,
        verbose: bool,
    ) -> Result<HotSpotPairRes, String> {
        const GENE_BATCH_SIZE: usize = 500;

        let gex_model =
            parse_gex_model(model).ok_or_else(|| format!("Invalid model type: {}", model))?;

        self.populate_umi_counts();

        let cell_set: IndexSet<u32> = self.cells_to_keep.iter().map(|&x| x as u32).collect();
        let reader = ParallelSparseReader::new(&self.f_path_gene).unwrap();

        let n_genes = gene_indices.len();
        let n_batches = n_genes.div_ceil(GENE_BATCH_SIZE);

        let mut lc_mat = Mat::zeros(n_genes, n_genes);
        let mut z_mat = Mat::zeros(n_genes, n_genes);

        if verbose {
            println!("Processing {} genes in {} batches", n_genes, n_batches);
        }

        for batch_i in 0..n_batches {
            let start_batch_i = Instant::now();

            let start_i = batch_i * GENE_BATCH_SIZE;
            let end_i = ((batch_i + 1) * GENE_BATCH_SIZE).min(n_genes);
            let batch_i_indices = &gene_indices[start_i..end_i];

            if verbose {
                println!(
                    "\nProcessing batch {} / {} (genes {}-{})",
                    batch_i + 1,
                    n_batches,
                    start_i,
                    end_i - 1
                );
            }

            let mut batch_i_chunks = reader.read_gene_parallel(batch_i_indices);
            batch_i_chunks.par_iter_mut().for_each(|chunk| {
                chunk.filter_selected_cells(&cell_set);
            });

            let batch_i_centered: Vec<Vec<f32>> = batch_i_chunks
                .par_iter()
                .map(|gene| {
                    create_centered_counts_gene(
                        gene,
                        self.umi_counts.as_ref().unwrap(),
                        self.n_cells,
                        &gex_model,
                    )
                })
                .collect();

            let batch_i_eg2: Vec<f32> = batch_i_centered
                .par_iter()
                .map(|counts| conditional_eg2(counts, self.neighbours, &self.weights))
                .collect();

            let end_batch_i = start_batch_i.elapsed();

            if verbose {
                println!("Computed batch {} in: {:.2?}", batch_i + 1, end_batch_i);
            }

            let remaining_batches = n_batches - batch_i;
            if verbose {
                println!(
                    "  Computing pairs with {} remaining batches",
                    remaining_batches
                );
            }

            let start_remaining_batches = Instant::now();

            for batch_j in batch_i..n_batches {
                let start_j = batch_j * GENE_BATCH_SIZE;
                let end_j = ((batch_j + 1) * GENE_BATCH_SIZE).min(n_genes);

                if verbose {
                    println!("    Batch pair ({}, {})", batch_i + 1, batch_j + 1);
                }

                let (batch_j_centered, batch_j_eg2) = if batch_i == batch_j {
                    (batch_i_centered.clone(), batch_i_eg2.clone())
                } else {
                    let batch_j_indices = &gene_indices[start_j..end_j];
                    let mut batch_j_chunks = reader.read_gene_parallel(batch_j_indices);
                    batch_j_chunks.par_iter_mut().for_each(|chunk| {
                        chunk.filter_selected_cells(&cell_set);
                    });

                    let centered: Vec<Vec<f32>> = batch_j_chunks
                        .par_iter()
                        .map(|gene| {
                            create_centered_counts_gene(
                                gene,
                                self.umi_counts.as_ref().unwrap(),
                                self.n_cells,
                                &gex_model,
                            )
                        })
                        .collect();

                    let eg2: Vec<f32> = centered
                        .par_iter()
                        .map(|counts| conditional_eg2(counts, self.neighbours, &self.weights))
                        .collect();

                    (centered, eg2)
                };

                let pairs: Vec<(usize, usize)> = if batch_i == batch_j {
                    (0..batch_i_centered.len())
                        .flat_map(|i| ((i + 1)..batch_i_centered.len()).map(move |j| (i, j)))
                        .collect()
                } else {
                    (0..batch_i_centered.len())
                        .flat_map(|i| (0..batch_j_centered.len()).map(move |j| (i, j)))
                        .collect()
                };

                let results: Vec<(usize, usize, f32, f32, f32)> = pairs
                    .par_iter()
                    .map(|&(local_i, local_j)| {
                        let x = &batch_i_centered[local_i];
                        let y = &batch_j_centered[local_j];

                        let lc = local_cov_pair_vec(x, y, self.neighbours, &self.weights) * 2.0;

                        let max_i = compute_local_cov_max(&self.node_degrees, x);
                        let max_j = compute_local_cov_max(&self.node_degrees, y);
                        let lc_max = (max_i + max_j) / 2.0;

                        let eg = 0.0;
                        let stdg_xy = batch_i_eg2[local_i].sqrt();
                        let z_xy = (lc - eg) / stdg_xy;

                        let stdg_yx = batch_j_eg2[local_j].sqrt();
                        let z_yx = (lc - eg) / stdg_yx;

                        let z = if z_xy.abs() < z_yx.abs() { z_xy } else { z_yx };

                        let global_i = start_i + local_i;
                        let global_j = start_j + local_j;

                        (global_i, global_j, lc, z, lc_max)
                    })
                    .collect();

                for (i, j, lc, z, lc_max) in results {
                    if !z.is_finite() {
                        continue;
                    }
                    let normalised_lc = if lc_max > 0.0 { lc / lc_max } else { 0.0 };

                    lc_mat[(i, j)] = normalised_lc;
                    lc_mat[(j, i)] = normalised_lc;
                    z_mat[(i, j)] = z;
                    z_mat[(j, i)] = z;
                }
            }

            let end_remaining_batches = start_remaining_batches.elapsed();

            if verbose {
                println!(
                    "Calculated all batches for batch {} in: {:.2?}",
                    batch_i + 1,
                    end_remaining_batches
                );
            }
        }

        if verbose {
            println!("Done!");
        }

        Ok(HotSpotPairRes {
            cor: lc_mat,
            z_scores: z_mat,
        })
    }

    /// Helper function to get the UMI counts per cell
    ///
    /// Reads and caches the total UMI count per cell for use in statistical
    /// models.
    fn populate_umi_counts(&mut self) {
        let reader = ParallelSparseReader::new(&self.f_path_cell).unwrap();
        let lib_sizes = reader.read_cell_library_sizes(self.cells_to_keep);
        let umi_counts = lib_sizes.iter().map(|x| *x as f32).collect::<Vec<f32>>();
        self.umi_counts = Some(umi_counts);
    }
}

/// Cluster the HotSpot Z matrix
///
/// ### Params
///
/// * `z_mat` - Reference to the Z-score matrix.
/// * `fdr_threshold` - Below which FDR to not cluster the genes anymore.
/// * `min_cluster_genes` - Minimum number of genes per cluster
///
/// ### Returns
///
/// The gene to cluster assignment. `NA`s indicate now assignment.
pub fn hotspot_gene_clusters(
    z_mat: MatRef<f64>,
    fdr_threshold: f64,
    min_cluster_genes: usize,
) -> Vec<f64> {
    let z_upper_triangle = faer_mat_to_upper_triangle(z_mat, 1);
    let pvals = z_scores_to_pval(&z_upper_triangle, "twosided");
    let fdrs = calc_fdr(&pvals);
    let n = z_mat.nrows();

    let z_threshold = z_upper_triangle
        .iter()
        .zip(fdrs.iter())
        .filter(|&(_, &fdr)| fdr < fdr_threshold)
        .map(|(&z, _)| z.abs())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(f64::INFINITY);

    let mut z = z_mat.to_owned();

    let mut active: Vec<usize> = (0..n).collect();
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut sizes: Vec<usize> = vec![1; n];

    // Labels: None if unlabelled, Some(label_id) if labelled
    let mut labels: Vec<Option<usize>> = vec![None; n];
    let mut next_label = 0usize;

    let mut gene_to_cluster: Vec<usize> = (0..n).collect();

    while active.len() > 1 {
        let mut max_z = f64::NEG_INFINITY;
        let mut max_i = 0;
        let mut max_j = 0;

        for (ai, &i) in active.iter().enumerate() {
            for &j in active.iter().skip(ai + 1) {
                if z[(i, j)] > max_z {
                    max_z = z[(i, j)];
                    max_i = i;
                    max_j = j;
                }
            }
        }

        if max_z < z_threshold {
            break;
        }

        let new_size = sizes[max_i] + sizes[max_j];
        let both_labelled = labels[max_i].is_some() && labels[max_j].is_some();

        for &k in &active {
            if k != max_i && k != max_j {
                let new_z = (z[(max_i, k)] * sizes[max_i] as f64
                    + z[(max_j, k)] * sizes[max_j] as f64)
                    / new_size as f64;
                z[(max_i, k)] = new_z;
                z[(k, max_i)] = new_z;
            }
        }

        let j_genes = clusters[max_j].drain(..).collect::<Vec<_>>();
        clusters[max_i].extend(j_genes);
        sizes[max_i] = new_size;

        for &gene in &clusters[max_i] {
            gene_to_cluster[gene] = max_i;
        }

        if new_size >= min_cluster_genes && !both_labelled {
            labels[max_i] = Some(next_label);
            next_label += 1;
        } else if both_labelled {
            labels[max_i] = None;
        }

        active.retain(|&x| x != max_j);
    }

    let mut gene_labels = vec![f64::NAN; n];
    for gene in 0..n {
        let cluster = gene_to_cluster[gene];
        if let Some(label) = labels[cluster] {
            gene_labels[gene] = label as f64;
        }
    }

    gene_labels
}
