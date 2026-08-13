//! Covariance kernels for the Gaussian process models.
//!
//! Only Matérn-5/2 for now, and deliberately as two free functions rather than
//! a trait or an enum. A `dyn Fn` costs a virtual call on every one of the tens
//! of millions of elements a fit evaluates, and a generic `K: Kernel` bound
//! would propagate through every signature and every error path to support a
//! single kernel. When a second one arrives, turn the fill into an enum matched
//! once per call, which is free in the inner loop.
//!
//! The fills are 1-D only. That is what the trajectory work needs, and for
//! scalar inputs the distance is `|x - x'|`, so fusing distance and kernel into
//! one write avoids materialising a separate distance matrix entirely.

use faer::MatMut;

/// `sqrt(5)`, the Matérn-5/2 scaling constant.
const SQRT_5: f64 = 2.236_067_977_499_79;

/// Matérn-5/2 covariance at a scalar distance.
///
/// `k(d) = (1 + r + r² / 3) exp(-r)` with `r = sqrt(5) d / ls`, the standard
/// form of `(1 + sqrt(5)d/l + 5d²/(3l²)) exp(-sqrt(5)d/l)`.
///
/// Twice differentiable, so the posterior mean is smooth without being as
/// aggressively so as a squared exponential. This is mellon's default and
/// therefore what Palantir's gene trends are fitted with.
///
/// ### Params
///
/// * `dist` - Distance between the two points, non-negative.
/// * `length_scale` - Length scale of the kernel, strictly positive.
///
/// ### Returns
///
/// The covariance, in `(0, 1]`, exactly `1` at zero distance.
///
/// ### References
///
/// Rasmussen and Williams, Gaussian Processes for Machine Learning, 2006,
/// equation 4.17.
#[inline]
pub fn matern52(dist: f64, length_scale: f64) -> f64 {
    let r = SQRT_5 * dist / length_scale;
    (1.0 + r + r * r / 3.0) * (-r).exp()
}

/// Fill a cross-covariance matrix for 1-D inputs.
///
/// `dst[(i, j)] = matern52(|a_i - b_j|, length_scale)`. Written column by
/// column to match faer's column-major storage.
///
/// mellon adds `1e-12` to the *squared* distance before its `sqrt`
/// (`mellon/util.py`), so its self-distance is `1e-6` rather than zero and its
/// diagonal comes out slightly below one: `1 - 8.3e-13` at `ls = 1.0`, growing
/// to `1 - 3.7e-11` at `ls = 0.15`, since the offset is absolute while the
/// kernel argument scales with `1 / ls`. We take the absolute difference
/// directly, so ours is exactly one. The gap sits far below any jitter that
/// would be added on top.
///
/// ### Params
///
/// * `dst` - Destination, `a.len()` by `b.len()`, fully overwritten.
/// * `a` - Row coordinates.
/// * `b` - Column coordinates.
/// * `length_scale` - Length scale of the kernel, strictly positive.
///
/// ### Panics
///
/// If `dst` does not have shape `(a.len(), b.len())`.
pub fn fill_matern52_cross_1d(mut dst: MatMut<'_, f64>, a: &[f64], b: &[f64], length_scale: f64) {
    assert_eq!(dst.nrows(), a.len(), "kernel fill: row count mismatch");
    assert_eq!(dst.ncols(), b.len(), "kernel fill: column count mismatch");

    for (j, &b_j) in b.iter().enumerate() {
        for (i, &a_i) in a.iter().enumerate() {
            dst[(i, j)] = matern52((a_i - b_j).abs(), length_scale);
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::Mat;

    /// Zero distance is unit covariance whatever the length scale.
    #[test]
    fn test_matern52_unit_at_zero() {
        for ls in [0.1, 1.0, 7.5] {
            assert_relative_eq!(matern52(0.0, ls), 1.0, epsilon = 1e-15);
        }
    }

    /// Against the unexpanded textbook form, which is written differently
    /// enough that a transcription slip in either would show up.
    #[test]
    fn test_matern52_matches_textbook_form() {
        let ls = 1.3_f64;
        for d in [0.0_f64, 0.25, 0.5, 1.0, 2.0, 5.0] {
            let want =
                (1.0 + SQRT_5 * d / ls + 5.0 * d * d / (3.0 * ls * ls)) * (-SQRT_5 * d / ls).exp();
            assert_relative_eq!(matern52(d, ls), want, epsilon = 1e-14, max_relative = 1e-14);
        }
    }

    /// A valid covariance decays monotonically; the growing polynomial must not win.
    #[test]
    fn test_matern52_is_monotone_decreasing() {
        let ls = 0.8_f64;
        let mut prev = matern52(0.0, ls);
        for step in 1..200 {
            let cur = matern52(step as f64 * 0.05, ls);
            assert!(cur < prev, "not decreasing at step {}", step);
            prev = cur;
        }
    }

    /// The length scale must be the only thing setting the decay rate: `k` at
    /// distance `d` with scale `ls` equals `k` at `2d` with scale `2 ls`.
    #[test]
    fn test_matern52_scales_with_length_scale() {
        for d in [0.1_f64, 0.5, 3.0] {
            assert_relative_eq!(
                matern52(d, 1.0),
                matern52(2.0 * d, 2.0),
                epsilon = 1e-14,
                max_relative = 1e-14
            );
        }
    }

    /// The fill must agree with the scalar kernel entry by entry, indices and all.
    #[test]
    fn test_fill_cross_matches_scalar_and_is_symmetric_on_self() {
        let a = vec![0.0_f64, 0.3, 0.9, 2.0];
        let b = vec![-1.0_f64, 0.3, 4.0];
        let ls = 1.1;

        let mut dst = Mat::<f64>::zeros(a.len(), b.len());
        fill_matern52_cross_1d(dst.as_mut(), &a, &b, ls);

        for (i, &a_i) in a.iter().enumerate() {
            for (j, &b_j) in b.iter().enumerate() {
                assert_relative_eq!(
                    dst[(i, j)],
                    matern52((a_i - b_j).abs(), ls),
                    epsilon = 1e-15
                );
            }
        }

        // A self-cross must come out symmetric with a unit diagonal.
        let mut gram = Mat::<f64>::zeros(a.len(), a.len());
        fill_matern52_cross_1d(gram.as_mut(), &a, &a, ls);
        for i in 0..a.len() {
            assert_relative_eq!(gram[(i, i)], 1.0, epsilon = 1e-15);
            for j in 0..i {
                assert_relative_eq!(gram[(i, j)], gram[(j, i)], epsilon = 1e-15);
            }
        }
    }
}
