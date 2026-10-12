//! Pairwise column correlation and covariance on MLX.
//!
//! Mirrors [`crate::gpu::linalg::corr`]: Spearman ranks the columns on the
//! host, the matrix goes up feature-major (`[n_cols, n_rows]` row-major, i.e.
//! the column-major faer buffer as is), one Metal kernel centres and scales
//! every feature with `1/sqrt(n-1)` folded in, and one MLX `matmul` of that
//! against its own transpose is the result. f32 only.
//!
//! The column variance is two-pass (mean first, then the sum of squared
//! deviations), never `sumsq - n * mean^2`, which cancels badly in f32 once
//! the mean is large against the spread.

use std::borrow::Cow;
use std::time::{Duration, Instant};

use faer::{Mat, MatRef};

use crate::core::math::matrix_helpers::rank_matrix_col;
use crate::mlx::ffi::{Array, MLX_FLOAT32, MetalKernel, OutputSpec, Stream};
use crate::mlx::ffi::{clear_cache, eval_all, install_error_handler};
use crate::prelude::*;

///////////////
// Constants //
///////////////

/// Threads per threadgroup in the centre-scale kernel; one threadgroup per
/// feature, lanes stride the samples.
const CENTRE_SCALE_THREADS: i32 = 256;

/// Upper bound on SIMD groups per threadgroup (1024 threads / 32 lanes), sizes
/// the threadgroup scratch for the reduction.
const MAX_SIMD_GROUPS: u32 = 32;

/// Standard deviation below which a feature is treated as constant and left
/// unscaled. Same floor as the cubecl `column_stats` kernel.
const STD_EPS: f32 = 1e-10;

///////////
// Enums //
///////////

/// Which pairwise statistic [`column_pairwise_cor_mlx`] computes.
///
/// The MLX twin of the cubecl `GpuCorCov`, which lives behind the `gpu`
/// feature. Selected host-side; the kernel only sees whether to divide by the
/// standard deviation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MlxCorCov {
    /// Covariance. Columns are centred but not divided by their standard
    /// deviation.
    #[default]
    Covariance,
    /// Pearson correlation. Columns are centred and divided by their standard
    /// deviation.
    Pearson,
    /// Spearman correlation. Pearson over columns rank-transformed on the host.
    Spearman,
}

/////////////
// Structs //
/////////////

/// Wall time of each stage of one [`column_pairwise_cor_mlx_staged`] call.
#[derive(Clone, Copy, Debug, Default)]
pub struct MlxCorStages {
    /// Host side: Spearman ranks plus the feature-major flatten
    pub prep: Duration,
    /// Copy into an MLX array
    pub upload: Duration,
    /// Centre-scale kernel plus the Gram `matmul`, up to a blocking eval
    pub compute: Duration,
    /// Copy of the `[d, d]` result into the returned `Mat`
    pub readback: Duration,
}

impl MlxCorStages {
    /// Sum of all stages.
    ///
    /// ### Returns
    ///
    /// The end-to-end wall time
    pub fn total(&self) -> Duration {
        self.prep + self.upload + self.compute + self.readback
    }
}

////////////
// Kernel //
////////////

/// Threadgroup sum via `simd_sum` and one scratch slot per SIMD group. Every
/// thread gets the total.
const TG_SUM_HEADER: &str = r#"
inline float tg_sum(float v, threadgroup float* sh, uint sg, uint lane, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) sh[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 0.0f;
    for (uint k = 0; k < nsg; k++) r += sh[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}
"#;

/// One threadgroup per feature: two-pass mean and centred sum of squares,
/// then `out = (x - mean) * inv`, with `inv = 1/sqrt(n-1)` and, under
/// `SCALE_SD`, divided by the standard deviation unless it is below
/// `STD_EPS`. The feature row is read three times; it stays in cache for all
/// but the narrowest, longest shapes.
///
/// Inputs: `data` `[d, n]` f32, `n_rows` `[1]` u32. Output: `out` `[d, n]`.
/// Template int: `SCALE_SD` (0 or 1).
const CENTRE_SCALE_SOURCE: &str = r#"
    uint feat = threadgroup_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;
    uint tpg = threads_per_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint nsg = simdgroups_per_threadgroup;
    uint n = n_rows[0];
    size_t base = size_t(feat) * n;
    threadgroup float sh[MAX_SIMD_GROUPS];

    float s = 0.0f;
    for (uint i = t; i < n; i += tpg) s += data[base + i];
    float mean = tg_sum(s, sh, sg, lane, nsg) / float(n);

    float ss = 0.0f;
    for (uint i = t; i < n; i += tpg) {
        float c = data[base + i] - mean;
        ss += c * c;
    }
    ss = tg_sum(ss, sh, sg, lane, nsg);

    float nm1 = float(n) - 1.0f;
    float inv = 1.0f / sqrt(nm1);
    if (SCALE_SD) {
        float sd = sqrt(ss / nm1);
        if (sd >= STD_EPS) inv /= sd;
    }
    for (uint i = t; i < n; i += tpg) out[base + i] = (data[base + i] - mean) * inv;
"#;

/////////////
// Helpers //
/////////////

/// The matrix as one flat feature-major buffer (`[n_cols, n_rows]`
/// row-major). Borrows the faer allocation when its columns sit back to
/// back, copies otherwise.
///
/// ### Params
///
/// * `mat` - Input matrix `[n_rows, n_cols]`
///
/// ### Returns
///
/// The `n_rows * n_cols` values, feature after feature
fn feature_major(mat: MatRef<'_, f32>) -> Cow<'_, [f32]> {
    let (n_rows, n_cols) = (mat.nrows(), mat.ncols());
    if let Some(cm) = mat.try_as_col_major()
        && cm.col_stride() as usize == n_rows
    {
        // SAFETY: unit row stride and no gap between columns, so the
        // `n_rows * n_cols` elements are one contiguous range from `as_ptr()`.
        return Cow::Borrowed(unsafe { std::slice::from_raw_parts(cm.as_ptr(), n_rows * n_cols) });
    }
    let mut buf = Vec::with_capacity(n_rows * n_cols);
    for j in 0..n_cols {
        buf.extend(mat.col(j).iter().copied());
    }
    Cow::Owned(buf)
}

/// Centre and scale every feature on the device, see [`CENTRE_SCALE_SOURCE`].
///
/// ### Params
///
/// * `data` - Feature-major input `[n_cols, n_rows]`
/// * `n_rows` - Samples per feature
/// * `n_cols` - Number of features
/// * `scale_sd` - Divide by the standard deviation (correlation) or not
///   (covariance)
/// * `s` - Stream to run on
///
/// ### Returns
///
/// The (lazy) scaled matrix `[n_cols, n_rows]`, `1/sqrt(n-1)` folded in
fn centre_scale(
    data: &Array,
    n_rows: i32,
    n_cols: i32,
    scale_sd: bool,
    s: &Stream,
) -> Result<Array, BixverseErrors> {
    let header = format!(
        "constant constexpr float STD_EPS = {STD_EPS:e}f;\n\
         constant constexpr uint MAX_SIMD_GROUPS = {MAX_SIMD_GROUPS};\n\
         {TG_SUM_HEADER}"
    );
    let kernel = MetalKernel::new(
        "bixverse_centre_scale",
        &["data", "n_rows"],
        &["out"],
        &header,
        CENTRE_SCALE_SOURCE,
    );
    let n_arr = Array::from_u32(&[n_rows as u32], &[1]);
    let mut out = kernel.apply(
        &[data, &n_arr],
        &[OutputSpec {
            shape: &[n_cols, n_rows],
            dtype: MLX_FLOAT32,
        }],
        [n_cols * CENTRE_SCALE_THREADS, 1, 1],
        [CENTRE_SCALE_THREADS, 1, 1],
        &[("SCALE_SD", scale_sd as i32)],
        s,
    )?;
    Ok(out.remove(0))
}

//////////
// Main //
//////////

/// Pairwise column correlation or covariance on MLX, with a per-stage split.
///
/// Same work as [`column_pairwise_cor_mlx`]. There is exactly one blocking
/// eval (after the Gram), since the upload is a synchronous host copy, so the
/// split costs nothing extra.
///
/// ### Params
///
/// * `mat` - Input matrix `[n_rows, n_cols]`, samples by features
/// * `cor_type` - Correlation or covariance variant to compute
///
/// ### Returns
///
/// The `[n_cols, n_cols]` matrix and the stage timings
///
/// ### Errors
///
/// * `InvalidArgument` if `n_rows < 2` or a dimension exceeds what an MLX
///   shape or launch grid can index
/// * `MlxError` if mlx-c reports a failure
pub fn column_pairwise_cor_mlx_staged(
    mat: MatRef<f32>,
    cor_type: MlxCorCov,
) -> Result<(Mat<f32>, MlxCorStages), BixverseErrors> {
    let (n_rows, n_cols) = (mat.nrows(), mat.ncols());
    if n_rows < 2 {
        return Err(BixverseErrors::InvalidArgument(format!(
            "MLX correlation needs at least 2 rows, got {n_rows}"
        )));
    }
    let max_cols = (i32::MAX / CENTRE_SCALE_THREADS) as usize;
    if n_rows > i32::MAX as usize || n_cols > max_cols {
        return Err(BixverseErrors::InvalidArgument(format!(
            "MLX correlation: {n_rows} x {n_cols} exceeds the MLX index range \
             ({} rows, {max_cols} columns)",
            i32::MAX
        )));
    }
    let (n_i, d_i) = (n_rows as i32, n_cols as i32);
    let mut st = MlxCorStages::default();
    install_error_handler();
    let s = Stream::default_gpu();

    let t = Instant::now();
    let ranked = matches!(cor_type, MlxCorCov::Spearman).then(|| rank_matrix_col(&mat));
    let mat = ranked.as_ref().map(|m| m.as_ref()).unwrap_or(mat);
    let flat = feature_major(mat);
    st.prep = t.elapsed();

    let t = Instant::now();
    let data = Array::from_f32(&flat, &[d_i, n_i]);
    st.upload = t.elapsed();

    let t = Instant::now();
    let scale_sd = !matches!(cor_type, MlxCorCov::Covariance);
    let scaled = centre_scale(&data, n_i, d_i, scale_sd, &s)?;
    let gram = scaled.matmul(&scaled.transpose(&s)?, &s)?;
    eval_all(&[&gram], false)?;
    st.compute = t.elapsed();

    // Row-major and symmetric, so read the transpose: sequential walk into
    // faer's column-major fill.
    let t = Instant::now();
    let g = gram.as_f32()?;
    let out = Mat::from_fn(n_cols, n_cols, |i, j| g[j * n_cols + i]);
    st.readback = t.elapsed();

    drop((gram, scaled, data));
    clear_cache()?;
    Ok((out, st))
}

/// Pairwise column correlation or covariance on MLX.
///
/// For Spearman the columns are rank-transformed on the host first. The
/// matrix is then centred and scaled on the device, `1/sqrt(n-1)` folded in,
/// so a single `S Sᵀ` over the feature-major `S` is the result.
///
/// ### Params
///
/// * `mat` - Input matrix `[n_rows, n_cols]`, samples by features
/// * `cor_type` - Correlation or covariance variant to compute
/// * `verbose` - Print the stage timings to stdout
///
/// ### Returns
///
/// Pairwise column correlation or covariance matrix `[n_cols, n_cols]`
///
/// ### Errors
///
/// * `InvalidArgument` if `n_rows < 2` or a dimension exceeds what an MLX
///   shape or launch grid can index
/// * `MlxError` if mlx-c reports a failure
pub fn column_pairwise_cor_mlx(
    mat: MatRef<f32>,
    cor_type: MlxCorCov,
    verbose: bool,
) -> Result<Mat<f32>, BixverseErrors> {
    let (out, st) = column_pairwise_cor_mlx_staged(mat, cor_type)?;
    if verbose {
        println!(
            "MLX correlation: prep {:.2?}, upload {:.2?}, compute {:.2?}, readback {:.2?}, \
             total {:.2?}",
            st.prep,
            st.upload,
            st.compute,
            st.readback,
            st.total()
        );
    }
    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::base::cors_similarity::{column_pairwise_cor, column_pairwise_cov};

    /// Relative tolerance against the reference maximum, as in
    /// `tests/gpu_corr.rs`.
    const REL_TOL: f32 = 1e-3;

    /// Deterministic hash to `[0, 1)`.
    fn hash01(a: usize, b: usize) -> f32 {
        let mut h = (a as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
        h ^= h >> 29;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 32;
        ((h >> 40) as f32) / 16_777_216.0
    }

    /// `[n, d]` noise plus a shared factor and a per-column offset, so the
    /// correlations are not near zero and the means are large against the
    /// spread.
    fn make_data(n: usize, d: usize, offset: f32) -> Mat<f32> {
        Mat::from_fn(n, d, |i, j| {
            offset * (j as f32 + 1.0)
                + (hash01(i, 7) - 0.5) * (hash01(j, 11) - 0.5)
                + hash01(i, j + 1000)
                - 0.5
        })
    }

    /// The CPU reference, computed in f64 and cast down.
    fn cpu_ref(mat: &Mat<f32>, cor_type: MlxCorCov) -> Mat<f32> {
        let m64 = Mat::from_fn(mat.nrows(), mat.ncols(), |i, j| mat[(i, j)] as f64);
        let r = match cor_type {
            MlxCorCov::Covariance => column_pairwise_cov(&m64.as_ref()),
            MlxCorCov::Pearson => column_pairwise_cor(&m64.as_ref(), false),
            MlxCorCov::Spearman => column_pairwise_cor(&m64.as_ref(), true),
        };
        Mat::from_fn(r.nrows(), r.ncols(), |i, j| r[(i, j)] as f32)
    }

    /// Run MLX against the CPU reference and assert the max abs difference is
    /// within `REL_TOL` of the reference maximum (floored at one).
    fn assert_matches_cpu(mat: &Mat<f32>, cor_type: MlxCorCov, label: &str) {
        let got = column_pairwise_cor_mlx(mat.as_ref(), cor_type, false).unwrap();
        let want = cpu_ref(mat, cor_type);
        let (mut max_diff, mut want_max) = (0.0f32, 0.0f32);
        for i in 0..want.nrows() {
            for j in 0..want.ncols() {
                want_max = want_max.max(want[(i, j)].abs());
                max_diff = max_diff.max((got[(i, j)] - want[(i, j)]).abs());
            }
        }
        eprintln!("{label}: max abs diff {max_diff:.3e} (ref max {want_max:.3e})");
        assert!(
            max_diff <= REL_TOL * want_max.max(1.0),
            "{label}: MLX and CPU disagree by {max_diff:.3e} against a max of {want_max:.3e}"
        );
    }

    #[test]
    fn test_mlx_pearson_matches_cpu() {
        assert_matches_cpu(&make_data(500, 64, 0.0), MlxCorCov::Pearson, "pearson");
    }

    #[test]
    fn test_mlx_covariance_matches_cpu() {
        assert_matches_cpu(&make_data(500, 64, 0.0), MlxCorCov::Covariance, "cov");
    }

    /// Means of 1e3 to 3e4 against a spread under one: a one-pass variance
    /// would cancel to garbage in f32.
    #[test]
    fn test_mlx_large_mean_no_cancellation() {
        let mat = make_data(2_000, 30, 1_000.0);
        assert_matches_cpu(&mat, MlxCorCov::Pearson, "pearson offset");
        assert_matches_cpu(&mat, MlxCorCov::Covariance, "cov offset");
    }

    /// Quantised values, so the ranks carry ties.
    #[test]
    fn test_mlx_spearman_matches_cpu() {
        let mat = Mat::from_fn(300, 40, |i, j| {
            (hash01(i, j) * 8.0).floor() + (i % 5) as f32
        });
        assert_matches_cpu(&mat, MlxCorCov::Spearman, "spearman");
    }

    /// Fewer rows than one SIMD group, and sizes off every power of two.
    #[test]
    fn test_mlx_ragged_shapes() {
        for &(n, d) in &[(5, 3), (37, 1), (1_001, 67), (257, 300)] {
            let mat = make_data(n, d, 2.0);
            for c in [
                MlxCorCov::Pearson,
                MlxCorCov::Covariance,
                MlxCorCov::Spearman,
            ] {
                assert_matches_cpu(&mat, c, &format!("{c:?} {n}x{d}"));
            }
        }
    }

    /// A strided submatrix takes the copying path of `feature_major`.
    #[test]
    fn test_mlx_submatrix_input() {
        let full = make_data(400, 50, 0.0);
        let sub = full.as_ref().submatrix(10, 5, 300, 20).to_owned();
        let got = column_pairwise_cor_mlx(
            full.as_ref().submatrix(10, 5, 300, 20),
            MlxCorCov::Pearson,
            false,
        )
        .unwrap();
        let want = column_pairwise_cor_mlx(sub.as_ref(), MlxCorCov::Pearson, false).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn test_mlx_pearson_unit_diagonal() {
        let got =
            column_pairwise_cor_mlx(make_data(800, 50, 3.0).as_ref(), MlxCorCov::Pearson, false)
                .unwrap();
        for j in 0..got.nrows() {
            assert!(
                (got[(j, j)] - 1.0).abs() < 1e-4,
                "diag[{j}] = {}",
                got[(j, j)]
            );
        }
    }

    #[test]
    fn test_mlx_output_symmetric() {
        let mat = make_data(300, 70, 1.0);
        for c in [
            MlxCorCov::Pearson,
            MlxCorCov::Covariance,
            MlxCorCov::Spearman,
        ] {
            let got = column_pairwise_cor_mlx(mat.as_ref(), c, false).unwrap();
            for i in 0..got.nrows() {
                for j in 0..i {
                    assert!(
                        (got[(i, j)] - got[(j, i)]).abs() < 1e-5,
                        "{c:?} not symmetric at ({i},{j}): {} vs {}",
                        got[(i, j)],
                        got[(j, i)]
                    );
                }
            }
        }
    }

    #[test]
    fn test_mlx_rejects_single_row() {
        let mat = Mat::<f32>::zeros(1, 4);
        assert!(column_pairwise_cor_mlx(mat.as_ref(), MlxCorCov::Pearson, false).is_err());
    }
}
