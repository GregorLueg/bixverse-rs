//! Grey-level co-occurrence matrices and Haralick features.
//!
//! Quantise a channel to a handful of grey levels, count how often each pair of
//! levels sits at a given offset from each other, then reduce that matrix to
//! five numbers.
//!
//! Two choices worth knowing about. The matrix is accumulated over all four
//! offsets before it gets normalised, which makes the result rotation-invariant
//! rather than producing four sets of features to average afterwards. And it is
//! symmetric: every pair is counted in both directions, so the marginals are
//! equal and correlation only needs one of them.
//!
//! Run this on the haematoxylin channel as well as on grey. For anything
//! nuclear, tertiary lymphoid structures being the case in point, haematoxylin
//! carries the density signal and grey dilutes it with whatever eosin is doing.
//!
//! One caveat on tile size. GLCM wants enough pixels to fill `levels^2` bins.
//! At the default 32 levels that is 1024 bins, and a Visium spot in the hires
//! PNG is 14.4 px across, i.e. about 200 pixels. The features come out of that
//! but they are close to noise. The shipped CytAssist frame only gets you to
//! 21.6 px. Drop `glcm_levels` to 8 or 16 at those sizes, or work off the
//! full-res image (256 px per spot) if the texture numbers need to mean
//! something.
//!
//! ### References
//!
//! Haralick, Shanmugam & Dinstein, Textural features for image classification,
//! IEEE Trans Syst Man Cybern, 1973

use crate::prelude::*;

///////////////
// Constants //
///////////////

/// Default number of grey levels for quantisation.
pub const DEFAULT_GLCM_LEVELS: u8 = 32;

/// Default offsets at distance 1, as `(dy, dx)`: 0, 90, 45 and 135 degrees.
pub const DEFAULT_GLCM_OFFSETS: [(i8, i8); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

/// Number of Haralick features produced per channel.
pub const N_HARALICK_FEATURES: usize = 5;

/// Fixed upper bound for stain-concentration quantisation, in optical density.
///
/// Stain channels have no natural 0-255 range, so the bin edges have to be
/// pinned to something fixed rather than to each tile's own min and max.
/// Per-tile ranges would make the features incomparable between spots, which
/// defeats the purpose. Saturated haematoxylin sits around 2.0 OD, so 3.0
/// leaves headroom and clips only genuinely extreme pixels.
pub const STAIN_QUANT_MAX: f32 = 3.0;

//////////////
// Features //
//////////////

/// The five Haralick reductions of a co-occurrence matrix.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HaralickFeatures {
    /// Sum of `p(i,j) * (i - j)^2`. Local intensity variation, 0 for a flat
    /// tile.
    pub contrast: f32,
    /// Sum of `p(i,j) / (1 + (i - j)^2)`. High when neighbours look alike.
    pub homogeneity: f32,
    /// Linear dependency between neighbouring levels, in `[-1, 1]`. Defined as
    /// 1.0 for a constant tile, where the variance is zero.
    pub correlation: f32,
    /// Angular second moment, `sum p(i,j)^2`. High when few pairs dominate.
    pub energy: f32,
    /// Shannon entropy of the matrix in bits. High when the pairs are spread
    /// thin.
    pub entropy: f32,
}

impl HaralickFeatures {
    /// The features in a fixed order, matching [`HARALICK_FEATURE_NAMES`].
    ///
    /// ### Returns
    ///
    /// `[contrast, homogeneity, correlation, energy, entropy]`.
    pub fn to_array(self) -> [f32; N_HARALICK_FEATURES] {
        [
            self.contrast,
            self.homogeneity,
            self.correlation,
            self.energy,
            self.entropy,
        ]
    }
}

/// Suffixes for the Haralick features, in [`HaralickFeatures::to_array`] order.
pub const HARALICK_FEATURE_NAMES: [&str; N_HARALICK_FEATURES] = [
    "contrast",
    "homogeneity",
    "correlation",
    "energy",
    "entropy",
];

//////////////////
// Quantisation //
//////////////////

/// Quantise 8-bit values into `levels` bins.
///
/// ### Params
///
/// * `values` - Source bytes.
/// * `levels` - Number of bins, 2 to 255.
/// * `out` - Output buffer, cleared and refilled.
pub fn quantise_u8(values: &[u8], levels: u8, out: &mut Vec<u8>) {
    let levels = u16::from(levels);
    out.clear();
    out.reserve(values.len());
    out.extend(
        values
            .iter()
            .map(|&v| ((u16::from(v) * levels) / 256).min(levels - 1) as u8),
    );
}

/// Quantise stain concentrations into `levels` bins over `[0, STAIN_QUANT_MAX]`.
///
/// Values outside the range clip rather than rescale, so the bin edges stay put
/// across tiles. Negative concentrations happen where the linear stain model
/// does not hold, e.g. specular highlights, and land in bin 0.
///
/// ### Params
///
/// * `values` - Stain concentrations in optical density.
/// * `levels` - Number of bins, 2 to 255.
/// * `out` - Output buffer, cleared and refilled.
pub fn quantise_stain(values: &[f32], levels: u8, out: &mut Vec<u8>) {
    let n_levels = f32::from(levels);
    let top = levels - 1;
    out.clear();
    out.reserve(values.len());
    out.extend(values.iter().map(|&v| {
        let scaled = (v / STAIN_QUANT_MAX * n_levels).floor();
        if scaled <= 0.0 {
            0
        } else if scaled >= f32::from(top) {
            top
        } else {
            scaled as u8
        }
    }));
}

//////////
// GLCM //
//////////

/// Reusable scratch for the co-occurrence pass.
///
/// A `levels^2` count buffer plus a quantisation buffer. Held per Rayon thread
/// via `map_init` so the per-spot loop allocates nothing.
#[derive(Debug, Clone, Default)]
pub struct GlcmScratch {
    /// `levels * levels` counts, row-major.
    counts: Vec<f64>,
    /// Quantised channel, one byte per pixel.
    pub quantised: Vec<u8>,
}

impl GlcmScratch {
    /// Fresh scratch. Buffers size themselves on first use.
    ///
    /// ### Returns
    ///
    /// An empty scratch.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Build a symmetric normalised GLCM and reduce it to Haralick features.
///
/// ### Params
///
/// * `quantised` - Row-major quantised channel, values in `0..levels`.
/// * `width` - Image width in pixels.
/// * `height` - Image height in pixels.
/// * `levels` - Number of grey levels, 2 to 255.
/// * `offsets` - Neighbour offsets as `(dy, dx)`. All are accumulated into one
///   matrix.
/// * `scratch` - Reusable count buffer.
///
/// ### Returns
///
/// The features. A tile too small for any offset to fit yields
/// [`HaralickFeatures::default`] (all zero) rather than NaN, because that is a
/// degenerate tile rather than an error.
pub fn glcm_haralick(
    quantised: &[u8],
    width: u32,
    height: u32,
    levels: u8,
    offsets: &[(i8, i8)],
    scratch: &mut GlcmScratch,
) -> Result<HaralickFeatures, BixverseErrors> {
    if levels < 2 {
        return Err(BixverseErrors::GlcmLevelsOutOfRange(levels));
    }
    if offsets.is_empty() {
        return Err(BixverseErrors::GlcmNoOffsets);
    }
    // One scan up front beats an out-of-bounds index inside the hot loop, and
    // it is cheap next to the `offsets.len()` passes that follow.
    if let Some(&max) = quantised.iter().max()
        && max >= levels
    {
        return Err(BixverseErrors::GlcmValueOutOfRange { value: max, levels });
    }

    let n_levels = levels as usize;
    scratch.counts.clear();
    scratch.counts.resize(n_levels * n_levels, 0.0);

    let (w, h) = (width as i64, height as i64);
    let mut total = 0.0_f64;

    for &(dy, dx) in offsets {
        let (dy, dx) = (i64::from(dy), i64::from(dx));
        // Rows where both y and y + dy land inside the tile.
        let y_start = (-dy).max(0);
        let y_end = (h - dy).min(h);
        let x_start = (-dx).max(0);
        let x_end = (w - dx).min(w);

        for y in y_start..y_end {
            for x in x_start..x_end {
                let i = quantised[(y * w + x) as usize] as usize;
                let j = quantised[((y + dy) * w + (x + dx)) as usize] as usize;
                // Symmetric: count the pair in both directions.
                scratch.counts[i * n_levels + j] += 1.0;
                scratch.counts[j * n_levels + i] += 1.0;
                total += 2.0;
            }
        }
    }

    if total == 0.0 {
        return Ok(HaralickFeatures::default());
    }

    Ok(reduce_glcm(&scratch.counts, n_levels, total))
}

/// Normalise a count matrix and compute the Haralick reductions.
///
/// The matrix is symmetric by construction, so the row and column marginals are
/// identical and only one is computed.
///
/// ### Params
///
/// * `counts` - `levels * levels` raw counts, row-major.
/// * `n_levels` - Matrix side.
/// * `total` - Sum of `counts`, strictly positive.
///
/// ### Returns
///
/// The five features.
fn reduce_glcm(counts: &[f64], n_levels: usize, total: f64) -> HaralickFeatures {
    let inv_total = 1.0 / total;

    let mut contrast = 0.0_f64;
    let mut homogeneity = 0.0_f64;
    let mut energy = 0.0_f64;
    let mut entropy = 0.0_f64;
    let mut marginal = vec![0.0_f64; n_levels];

    for i in 0..n_levels {
        for j in 0..n_levels {
            let p = counts[i * n_levels + j] * inv_total;
            if p == 0.0 {
                continue;
            }
            let d = i as f64 - j as f64;
            contrast += p * d * d;
            homogeneity += p / (1.0 + d * d);
            energy += p * p;
            entropy -= p * p.log2();
            marginal[i] += p;
        }
    }

    let mean: f64 = marginal
        .iter()
        .enumerate()
        .map(|(i, &p)| i as f64 * p)
        .sum();
    let variance: f64 = marginal
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let d = i as f64 - mean;
            d * d * p
        })
        .sum();

    // A constant tile has zero variance, so correlation is 0/0. Calling it
    // perfectly correlated is the usual convention and keeps the feature finite.
    let correlation = if variance <= 0.0 {
        1.0
    } else {
        let mut acc = 0.0_f64;
        for i in 0..n_levels {
            for j in 0..n_levels {
                let p = counts[i * n_levels + j] * inv_total;
                if p == 0.0 {
                    continue;
                }
                acc += p * (i as f64 - mean) * (j as f64 - mean);
            }
        }
        acc / variance
    };

    HaralickFeatures {
        contrast: contrast as f32,
        homogeneity: homogeneity as f32,
        correlation: correlation as f32,
        energy: energy as f32,
        entropy: entropy as f32,
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A 4x4 image, two grey levels, left half 0 and right half 1.
    ///
    /// Hand-computed for the single horizontal offset `(0, 1)`:
    ///
    /// Each of the 4 rows contributes the pairs (0,0), (0,1), (1,1), so across
    /// the tile there are 4 of each. Counting symmetrically gives
    /// `n00 = 8, n01 = 4, n10 = 4, n11 = 8`, total 24, hence
    /// `p00 = p11 = 1/3` and `p01 = p10 = 1/6`.
    ///
    /// * contrast    = 1/6 + 1/6                       = 1/3
    /// * homogeneity = 1/3 + 1/12 + 1/12 + 1/3         = 5/6
    /// * energy      = 2*(1/3)^2 + 2*(1/6)^2           = 5/18
    /// * entropy     = -2*(1/3)log2(1/3) - 2*(1/6)log2(1/6) = 1.9182958
    /// * marginals are (1/2, 1/2), so mean = 1/2 and variance = 1/4;
    ///   correlation = (4*(1/3)(1/4) - ... ) / (1/4)   = 1/3
    #[test]
    fn test_glcm_matches_hand_computed_4x4() {
        #[rustfmt::skip]
        let q: Vec<u8> = vec![
            0, 0, 1, 1,
            0, 0, 1, 1,
            0, 0, 1, 1,
            0, 0, 1, 1,
        ];
        let mut scratch = GlcmScratch::new();
        let f = glcm_haralick(&q, 4, 4, 2, &[(0, 1)], &mut scratch).unwrap();

        assert_relative_eq!(f.contrast, 1.0 / 3.0, epsilon = 1e-6);
        assert_relative_eq!(f.homogeneity, 5.0 / 6.0, epsilon = 1e-6);
        assert_relative_eq!(f.energy, 5.0 / 18.0, epsilon = 1e-6);
        assert_relative_eq!(f.entropy, 1.918_295_8, epsilon = 1e-5);
        assert_relative_eq!(f.correlation, 1.0 / 3.0, epsilon = 1e-6);
    }

    /// The same tile read vertically. Every vertical pair is like-with-like,
    /// so the matrix is diagonal: `p00 = p11 = 1/2`, no off-diagonal mass.
    #[test]
    fn test_glcm_vertical_offset_on_vertical_stripes() {
        #[rustfmt::skip]
        let q: Vec<u8> = vec![
            0, 0, 1, 1,
            0, 0, 1, 1,
            0, 0, 1, 1,
            0, 0, 1, 1,
        ];
        let mut scratch = GlcmScratch::new();
        let f = glcm_haralick(&q, 4, 4, 2, &[(1, 0)], &mut scratch).unwrap();

        assert_relative_eq!(f.contrast, 0.0, epsilon = 1e-6);
        assert_relative_eq!(f.homogeneity, 1.0, epsilon = 1e-6);
        assert_relative_eq!(f.energy, 0.5, epsilon = 1e-6);
        assert_relative_eq!(f.entropy, 1.0, epsilon = 1e-6);
        // Perfectly correlated: every neighbour matches.
        assert_relative_eq!(f.correlation, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_glcm_constant_tile_is_degenerate_but_finite() {
        let q = vec![3u8; 16];
        let mut scratch = GlcmScratch::new();
        let f = glcm_haralick(&q, 4, 4, 8, &DEFAULT_GLCM_OFFSETS, &mut scratch).unwrap();

        assert_relative_eq!(f.contrast, 0.0, epsilon = 1e-6);
        assert_relative_eq!(f.homogeneity, 1.0, epsilon = 1e-6);
        assert_relative_eq!(f.energy, 1.0, epsilon = 1e-6);
        assert_relative_eq!(f.entropy, 0.0, epsilon = 1e-6);
        assert_relative_eq!(f.correlation, 1.0, epsilon = 1e-6);
        assert!(f.to_array().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_glcm_checkerboard_is_maximally_contrasting() {
        #[rustfmt::skip]
        let q: Vec<u8> = vec![
            0, 1, 0, 1,
            1, 0, 1, 0,
            0, 1, 0, 1,
            1, 0, 1, 0,
        ];
        let mut scratch = GlcmScratch::new();
        // Horizontal only: every pair straddles the two levels.
        let f = glcm_haralick(&q, 4, 4, 2, &[(0, 1)], &mut scratch).unwrap();
        assert_relative_eq!(f.contrast, 1.0, epsilon = 1e-6);
        assert_relative_eq!(f.homogeneity, 0.5, epsilon = 1e-6);
        assert_relative_eq!(f.correlation, -1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_glcm_is_symmetric_under_offset_sign_flip() {
        #[rustfmt::skip]
        let q: Vec<u8> = vec![
            0, 1, 2, 3,
            1, 2, 3, 0,
            2, 3, 0, 1,
            3, 0, 1, 2,
        ];
        let mut scratch = GlcmScratch::new();
        let a = glcm_haralick(&q, 4, 4, 4, &[(0, 1)], &mut scratch).unwrap();
        let b = glcm_haralick(&q, 4, 4, 4, &[(0, -1)], &mut scratch).unwrap();
        for (x, y) in a.to_array().iter().zip(b.to_array().iter()) {
            assert_relative_eq!(x, y, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_glcm_tile_too_small_for_offset_returns_zeros() {
        // A single row cannot host a vertical pair.
        let q = vec![0u8, 1, 2, 3];
        let mut scratch = GlcmScratch::new();
        let f = glcm_haralick(&q, 4, 1, 4, &[(1, 0)], &mut scratch).unwrap();
        assert_eq!(f, HaralickFeatures::default());
    }

    #[test]
    fn test_glcm_rejects_bad_parameters() {
        let q = vec![0u8; 16];
        let mut scratch = GlcmScratch::new();
        assert!(matches!(
            glcm_haralick(&q, 4, 4, 1, &[(0, 1)], &mut scratch),
            Err(BixverseErrors::GlcmLevelsOutOfRange(1))
        ));
        assert!(matches!(
            glcm_haralick(&q, 4, 4, 8, &[], &mut scratch),
            Err(BixverseErrors::GlcmNoOffsets)
        ));
    }

    #[test]
    fn test_scratch_reuse_gives_identical_results() {
        let q: Vec<u8> = (0..64u8).map(|i| i % 5).collect();
        let small: Vec<u8> = q[..16].iter().map(|v| v % 3).collect();
        let mut scratch = GlcmScratch::new();
        let first = glcm_haralick(&q, 8, 8, 5, &DEFAULT_GLCM_OFFSETS, &mut scratch).unwrap();
        // A differently sized call in between must not leak counts.
        let _ = glcm_haralick(&small, 4, 4, 3, &[(0, 1)], &mut scratch).unwrap();
        let second = glcm_haralick(&q, 8, 8, 5, &DEFAULT_GLCM_OFFSETS, &mut scratch).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn test_glcm_rejects_unquantised_input() {
        // Values of 4 in a matrix sized for 3 levels used to index out of
        // bounds. Fail loudly instead.
        let q = vec![0u8, 1, 4, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let mut scratch = GlcmScratch::new();
        assert!(matches!(
            glcm_haralick(&q, 4, 4, 3, &[(0, 1)], &mut scratch),
            Err(BixverseErrors::GlcmValueOutOfRange {
                value: 4,
                levels: 3
            })
        ));
    }

    #[test]
    fn test_quantise_u8_spans_the_range() {
        let mut out = Vec::new();
        quantise_u8(&[0, 127, 128, 255], 2, &mut out);
        assert_eq!(out, vec![0, 0, 1, 1]);

        quantise_u8(&[0, 255], 32, &mut out);
        assert_eq!(out, vec![0, 31]);
    }

    #[test]
    fn test_quantise_stain_clips_at_both_ends() {
        let mut out = Vec::new();
        // Range is [0, 3.0) over 3 bins, so edges land at 1.0 and 2.0.
        quantise_stain(&[-1.0, 0.0, 0.5, 1.5, 2.5, 99.0], 3, &mut out);
        assert_eq!(out, vec![0, 0, 0, 1, 2, 2]);
    }
}
