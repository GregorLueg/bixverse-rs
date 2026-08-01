//! Colour deconvolution, Ruifrok-Johnston.
//!
//! Stains absorb light multiplicatively, so they only separate linearly in
//! optical density. Convert RGB to OD, then apply the inverse of the normalised
//! stain matrix to get one concentration per stain.
//!
//! Two stains are known for H&E, haematoxylin and eosin. The third basis vector
//! is their normalised cross product, which spans whatever is left. It carries
//! no biological meaning on its own, but without it the system is
//! underdetermined.
//!
//! The stain matrix is a parameter, not a constant, so it can be re-estimated
//! per slide later. Scanner and protocol both shift the vectors enough to
//! matter.
//!
//! ### References
//!
//! Ruifrok & Johnston, Quantification of histochemical staining by color
//! deconvolution, Anal Quant Cytol Histol, 2001

use crate::prelude::*;
use crate::spatial::sp_image::source::RgbTile;

///////////////
// Constants //
///////////////

/// Default haematoxylin vector in RGB, from Ruifrok & Johnston.
pub const RJ_HAEMATOXYLIN: [f64; 3] = [0.650, 0.704, 0.286];

/// Default eosin vector in RGB, from Ruifrok & Johnston.
pub const RJ_EOSIN: [f64; 3] = [0.072, 0.990, 0.105];

/// Offset in the optical-density transform.
///
/// `od = -log10((i + 1) / 256)` rather than `-log10(i / 255)`. Shifting by one
/// keeps a black pixel finite (OD 2.408 instead of infinity) and makes the
/// transform exactly invertible, which is what lets the round-trip test assert
/// something meaningful.
const OD_OFFSET: f64 = 1.0;

/// Denominator in the optical-density transform, `255 + OD_OFFSET`.
const OD_SCALE: f64 = 256.0;

/// Below this the stain matrix determinant counts as singular.
///
/// The matrix is built from unit vectors, so a well-conditioned basis has a
/// determinant of order 0.1 to 1. Anything at 1e-8 means two stain vectors have
/// collapsed onto each other.
const DET_EPSILON: f64 = 1e-8;

/////////////////
// StainMatrix //
/////////////////

/// A normalised stain basis plus its precomputed inverse.
///
/// Rows are stain vectors in RGB. Every row is unit length: the vectors set the
/// direction of absorption and the concentration carries the magnitude, so an
/// unnormalised row would just rescale its own channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StainMatrix {
    /// Unit stain vectors, row `s` is stain `s` in RGB.
    stains: [[f64; 3]; 3],
    /// Inverse of the transpose of `stains`. Concentration `s` is
    /// `sum_c deconv[s][c] * od[c]`.
    deconv: [[f64; 3]; 3],
}

impl StainMatrix {
    /// Build a stain matrix from three RGB absorption vectors.
    ///
    /// Rows get normalised to unit length here, so pass them in whatever scale
    /// is convenient. Any row that is all zeros, or a basis where two vectors
    /// are collinear, fails.
    ///
    /// ### Params
    ///
    /// * `stains` - Three RGB vectors, row per stain.
    ///
    /// ### Returns
    ///
    /// The matrix with its inverse cached, or
    /// [`BixverseErrors::StainMatrixSingular`] if the basis is degenerate.
    pub fn new(stains: [[f64; 3]; 3]) -> Result<Self, BixverseErrors> {
        let mut normalised = stains;
        for row in normalised.iter_mut() {
            let norm = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
            if norm < DET_EPSILON {
                return Err(BixverseErrors::StainMatrixSingular);
            }
            for v in row.iter_mut() {
                *v /= norm;
            }
        }

        // OD = V^T c, so c = (V^T)^-1 OD.
        let mut transposed = [[0.0_f64; 3]; 3];
        for (s, row) in normalised.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                transposed[c][s] = v;
            }
        }
        let deconv = invert_3x3(&transposed).ok_or(BixverseErrors::StainMatrixSingular)?;

        Ok(Self {
            stains: normalised,
            deconv,
        })
    }

    /// Build from haematoxylin and eosin alone.
    ///
    /// The third vector is the normalised cross product of the two, which
    /// completes the basis without claiming to be a stain.
    ///
    /// ### Params
    ///
    /// * `haem` - Haematoxylin vector in RGB.
    /// * `eosin` - Eosin vector in RGB.
    ///
    /// ### Returns
    ///
    /// The matrix, or [`BixverseErrors::StainMatrixSingular`] if the two
    /// vectors are collinear.
    pub fn from_two_stains(haem: [f64; 3], eosin: [f64; 3]) -> Result<Self, BixverseErrors> {
        let h = normalise(haem).ok_or(BixverseErrors::StainMatrixSingular)?;
        let e = normalise(eosin).ok_or(BixverseErrors::StainMatrixSingular)?;
        let residual = normalise([
            h[1] * e[2] - h[2] * e[1],
            h[2] * e[0] - h[0] * e[2],
            h[0] * e[1] - h[1] * e[0],
        ])
        .ok_or(BixverseErrors::StainMatrixSingular)?;

        Self::new([h, e, residual])
    }

    /// The unit stain vectors, row per stain.
    ///
    /// ### Returns
    ///
    /// The normalised basis.
    pub fn stains(&self) -> &[[f64; 3]; 3] {
        &self.stains
    }

    /// Deconvolve one pixel.
    ///
    /// ### Params
    ///
    /// * `rgb` - The pixel as `[r, g, b]`.
    ///
    /// ### Returns
    ///
    /// Concentrations as `[haematoxylin, eosin, residual]`, in the order the
    /// rows were supplied.
    #[inline]
    pub fn deconvolve_pixel(&self, rgb: [u8; 3]) -> [f64; 3] {
        self.deconvolve_od(rgb_to_od(rgb))
    }

    /// Deconvolve a pixel already in optical density.
    ///
    /// The linear half of the job, with no 8-bit quantisation anywhere near it.
    /// Exact inverse of [`StainMatrix::od_from_concentrations`].
    ///
    /// ### Params
    ///
    /// * `od` - Per-channel optical density.
    ///
    /// ### Returns
    ///
    /// Concentrations as `[haematoxylin, eosin, residual]`.
    #[inline]
    pub fn deconvolve_od(&self, od: [f64; 3]) -> [f64; 3] {
        let mut out = [0.0_f64; 3];
        for (s, row) in self.deconv.iter().enumerate() {
            out[s] = row[0] * od[0] + row[1] * od[1] + row[2] * od[2];
        }
        out
    }

    /// Optical density from stain concentrations.
    ///
    /// ### Params
    ///
    /// * `concentrations` - `[haematoxylin, eosin, residual]`.
    ///
    /// ### Returns
    ///
    /// Per-channel optical density.
    #[inline]
    pub fn od_from_concentrations(&self, concentrations: [f64; 3]) -> [f64; 3] {
        let mut od = [0.0_f64; 3];
        for (c, slot) in od.iter_mut().enumerate() {
            *slot = self.stains[0][c] * concentrations[0]
                + self.stains[1][c] * concentrations[1]
                + self.stains[2][c] * concentrations[2];
        }
        od
    }

    /// Reconstruct a pixel from stain concentrations.
    ///
    /// The exact inverse of [`StainMatrix::deconvolve_pixel`] up to the 8-bit
    /// rounding at the end. Mostly here so the round-trip can be tested.
    ///
    /// ### Params
    ///
    /// * `concentrations` - `[haematoxylin, eosin, residual]`.
    ///
    /// ### Returns
    ///
    /// The pixel as `[r, g, b]`.
    #[inline]
    pub fn reconstruct_pixel(&self, concentrations: [f64; 3]) -> [u8; 3] {
        od_to_rgb(self.od_from_concentrations(concentrations))
    }

    /// Deconvolve a whole tile into three caller-owned buffers.
    ///
    /// Takes the buffers rather than allocating so the per-spot driver can hold
    /// one set of scratch vectors per Rayon thread. They get cleared and
    /// resized to the tile's pixel count.
    ///
    /// ### Params
    ///
    /// * `tile` - Source tile.
    /// * `haem` - Output haematoxylin concentrations, one per pixel.
    /// * `eosin` - Output eosin concentrations, one per pixel.
    /// * `residual` - Output residual concentrations, one per pixel.
    pub fn deconvolve_tile(
        &self,
        tile: &RgbTile,
        haem: &mut Vec<f32>,
        eosin: &mut Vec<f32>,
        residual: &mut Vec<f32>,
    ) {
        let n = tile.n_pixels();
        haem.clear();
        eosin.clear();
        residual.clear();
        haem.reserve(n);
        eosin.reserve(n);
        residual.reserve(n);

        for px in tile.pixels() {
            let c = self.deconvolve_pixel([px[0], px[1], px[2]]);
            haem.push(c[0] as f32);
            eosin.push(c[1] as f32);
            residual.push(c[2] as f32);
        }
    }
}

impl Default for StainMatrix {
    /// The Ruifrok-Johnston H&E basis.
    fn default() -> Self {
        Self::from_two_stains(RJ_HAEMATOXYLIN, RJ_EOSIN)
            .expect("the Ruifrok-Johnston H&E vectors are non-collinear by construction")
    }
}

/////////////
// Helpers //
/////////////

/// Normalise a vector to unit length.
///
/// ### Params
///
/// * `v` - The vector.
///
/// ### Returns
///
/// The unit vector, or `None` if the input is (near) zero.
fn normalise(v: [f64; 3]) -> Option<[f64; 3]> {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm < DET_EPSILON {
        return None;
    }
    Some([v[0] / norm, v[1] / norm, v[2] / norm])
}

/// Invert a 3x3 matrix by the adjugate.
///
/// Faer would be overkill at this size and the closed form is exact.
///
/// ### Params
///
/// * `m` - The matrix, row-major.
///
/// ### Returns
///
/// The inverse, or `None` if the determinant is below [`DET_EPSILON`].
fn invert_3x3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;

    if det.abs() < DET_EPSILON {
        return None;
    }
    let inv_det = 1.0 / det;

    Some([
        [
            c00 * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            c01 * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            c02 * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

/// RGB to optical density.
///
/// ### Params
///
/// * `rgb` - The pixel.
///
/// ### Returns
///
/// Per-channel OD, in `[0, log10(256)]`.
#[inline]
pub fn rgb_to_od(rgb: [u8; 3]) -> [f64; 3] {
    let f = |c: u8| -((f64::from(c) + OD_OFFSET) / OD_SCALE).log10();
    [f(rgb[0]), f(rgb[1]), f(rgb[2])]
}

/// Optical density back to RGB.
///
/// The exact inverse of [`rgb_to_od`] before the cast to `u8`.
///
/// ### Params
///
/// * `od` - Per-channel optical density.
///
/// ### Returns
///
/// The pixel, clamped into `[0, 255]`.
#[inline]
pub fn od_to_rgb(od: [f64; 3]) -> [u8; 3] {
    let f = |d: f64| {
        let v = OD_SCALE * 10.0_f64.powf(-d) - OD_OFFSET;
        v.round().clamp(0.0, 255.0) as u8
    };
    [f(od[0]), f(od[1]), f(od[2])]
}

/// Mean optical density over every channel and pixel of a tile.
///
/// A blunt "how much stain is here at all" summary. Cheap and it separates
/// blank glass from tissue without any of the deconvolution machinery.
///
/// ### Params
///
/// * `tile` - The tile.
///
/// ### Returns
///
/// The mean OD, or 0.0 for an empty tile. Accumulated in `f64` because a
/// 256x256 tile is nearly 200k additions.
pub fn mean_optical_density(tile: &RgbTile) -> f32 {
    let n = tile.n_pixels();
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0.0_f64;
    for px in tile.pixels() {
        let od = rgb_to_od([px[0], px[1], px[2]]);
        acc += od[0] + od[1] + od[2];
    }
    (acc / (n as f64 * 3.0)) as f32
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_od_round_trips_every_byte() {
        for i in 0..=255u8 {
            let back = od_to_rgb(rgb_to_od([i, i, i]));
            assert_eq!(back, [i, i, i], "byte {i} did not round-trip");
        }
    }

    #[test]
    fn test_od_endpoints() {
        // White is zero absorbance, black saturates at log10(256).
        assert_relative_eq!(rgb_to_od([255, 255, 255])[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(rgb_to_od([0, 0, 0])[0], 256.0_f64.log10(), epsilon = 1e-12);
    }

    #[test]
    fn test_default_stain_rows_are_unit_length() {
        let m = StainMatrix::default();
        for row in m.stains() {
            let n = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
            assert_relative_eq!(n, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_residual_vector_is_orthogonal_to_both_stains() {
        let m = StainMatrix::default();
        let [h, e, r] = *m.stains();
        assert_relative_eq!(
            h[0] * r[0] + h[1] * r[1] + h[2] * r[2],
            0.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            e[0] * r[0] + e[1] * r[1] + e[2] * r[2],
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_invert_3x3_against_identity() {
        let m = [[2.0, 0.0, 1.0], [1.0, 3.0, 0.0], [0.0, 1.0, 4.0]];
        let inv = invert_3x3(&m).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let v: f64 = (0..3).map(|k| m[i][k] * inv[k][j]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(v, want, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_singular_stain_matrix_rejected() {
        // Two identical vectors span a plane, not a space.
        let res = StainMatrix::from_two_stains(RJ_HAEMATOXYLIN, RJ_HAEMATOXYLIN);
        assert!(matches!(res, Err(BixverseErrors::StainMatrixSingular)));

        let res = StainMatrix::new([[0.0, 0.0, 0.0], RJ_EOSIN, [0.0, 0.0, 1.0]]);
        assert!(matches!(res, Err(BixverseErrors::StainMatrixSingular)));
    }

    /// The headline check, in optical density where the maths is exact.
    ///
    /// A patch of pure haematoxylin at concentration `c` has OD `c * v_H`.
    /// Deconvolving it must return `[c, 0, 0]` to machine precision, because
    /// the deconvolution matrix is the exact inverse of the stain basis.
    #[test]
    fn test_pure_haematoxylin_patch_round_trips_exactly_in_od() {
        let m = StainMatrix::default();

        for &c_true in &[0.25_f64, 0.5, 1.0, 1.5, 2.0] {
            let od = m.od_from_concentrations([c_true, 0.0, 0.0]);
            let got = m.deconvolve_od(od);

            assert_relative_eq!(got[0], c_true, epsilon = 1e-12);
            assert_relative_eq!(got[1], 0.0, epsilon = 1e-12);
            assert_relative_eq!(got[2], 0.0, epsilon = 1e-12);
        }
    }

    /// The same patch pushed through an 8-bit image.
    ///
    /// The stain vectors are not orthogonal, so the inverse amplifies the
    /// half-LSB rounding of the RGB step. Error grows with concentration
    /// because darker pixels carry proportionally less information: at OD 1.5
    /// the eosin channel picks up about 0.013 of spurious signal. That is the
    /// cost of the format, not of the algorithm, and 0.02 OD is far below
    /// anything biologically readable.
    #[test]
    fn test_pure_haematoxylin_patch_round_trips_through_u8() {
        let m = StainMatrix::default();

        for &c_true in &[0.25_f64, 0.5, 1.0, 1.5] {
            let px = m.reconstruct_pixel([c_true, 0.0, 0.0]);
            let got = m.deconvolve_pixel(px);

            assert_relative_eq!(got[0], c_true, epsilon = 0.02);
            assert_relative_eq!(got[1], 0.0, epsilon = 0.02);
            assert_relative_eq!(got[2], 0.0, epsilon = 0.02);
        }
    }

    #[test]
    fn test_pure_eosin_patch_round_trips() {
        let m = StainMatrix::default();
        let got = m.deconvolve_od(m.od_from_concentrations([0.0, 0.8, 0.0]));
        assert_relative_eq!(got[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(got[1], 0.8, epsilon = 1e-12);
        assert_relative_eq!(got[2], 0.0, epsilon = 1e-12);

        let got = m.deconvolve_pixel(m.reconstruct_pixel([0.0, 0.8, 0.0]));
        assert_relative_eq!(got[1], 0.8, epsilon = 0.02);
    }

    #[test]
    fn test_mixed_stain_patch_round_trips() {
        let m = StainMatrix::default();
        let got = m.deconvolve_od(m.od_from_concentrations([0.7, 0.4, 0.15]));
        assert_relative_eq!(got[0], 0.7, epsilon = 1e-12);
        assert_relative_eq!(got[1], 0.4, epsilon = 1e-12);
        assert_relative_eq!(got[2], 0.15, epsilon = 1e-12);

        let got = m.deconvolve_pixel(m.reconstruct_pixel([0.7, 0.4, 0.0]));
        assert_relative_eq!(got[0], 0.7, epsilon = 0.02);
        assert_relative_eq!(got[1], 0.4, epsilon = 0.02);
    }

    #[test]
    fn test_white_pixel_has_no_stain() {
        let m = StainMatrix::default();
        let got = m.deconvolve_pixel([255, 255, 255]);
        for c in got {
            assert_relative_eq!(c, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_deconvolve_tile_matches_per_pixel() {
        let m = StainMatrix::default();
        let tile = RgbTile {
            data: vec![200, 150, 220, 90, 40, 110, 255, 255, 255, 10, 10, 10],
            width: 2,
            height: 2,
        };

        let (mut h, mut e, mut r) = (Vec::new(), Vec::new(), Vec::new());
        m.deconvolve_tile(&tile, &mut h, &mut e, &mut r);

        assert_eq!(h.len(), 4);
        for (i, px) in tile.pixels().enumerate() {
            let want = m.deconvolve_pixel([px[0], px[1], px[2]]);
            assert_relative_eq!(h[i], want[0] as f32, epsilon = 1e-6);
            assert_relative_eq!(e[i], want[1] as f32, epsilon = 1e-6);
            assert_relative_eq!(r[i], want[2] as f32, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_mean_optical_density_white_is_zero_black_is_max() {
        let white = RgbTile {
            data: vec![255; 12],
            width: 2,
            height: 2,
        };
        assert_relative_eq!(mean_optical_density(&white), 0.0, epsilon = 1e-6);

        let black = RgbTile {
            data: vec![0; 12],
            width: 2,
            height: 2,
        };
        assert_relative_eq!(
            mean_optical_density(&black),
            256.0_f64.log10() as f32,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_custom_stain_matrix_is_usable() {
        // A deliberately unnormalised basis: construction must fix it.
        let m = StainMatrix::new([[1.3, 1.408, 0.572], [0.144, 1.980, 0.210], [0.0, 0.0, 2.0]])
            .unwrap();
        for row in m.stains() {
            let n = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
            assert_relative_eq!(n, 1.0, epsilon = 1e-12);
        }
    }
}
