//! Which column of `obsm/spatial` is `x`.
//!
//! Nothing in the AnnData spec fixes the column order, so it gets resolved from
//! evidence in the file rather than assumed.
//!
//! ## What the order actually affects
//!
//! Swapping `x` and `y` is a reflection about the diagonal, and a reflection is
//! an isometry: every pairwise distance survives it. The kNN and radius graphs,
//! their weightings, Moran's I and the SPARK-X kernels are all built on those
//! distances, and the two lattice layouts run off the array indices instead. So
//! a transposed read leaves every spatial statistic in this crate **numerically
//! unchanged**.
//!
//! One thing does care: putting coordinates onto a histology image. Get it
//! wrong and `sp_image` cuts every tile from the transpose of where the spot
//! is. That is why the evidence below is ranked by how directly it measures
//! *that*, rather than by how authoritative it looks.
//!
//! ## Why the `obs` column names are not the top of the ladder
//!
//! `scanpy.read_visium` renames the tissue positions columns positionally:
//!
//! ```text
//! positions.columns = ["in_tissue", "array_row", "array_col",
//!                      "pxl_col_in_fullres", "pxl_row_in_fullres"]
//! ```
//!
//! Space Ranger writes those last two the other way round, so scanpy's
//! `pxl_row_in_fullres` holds the pixel *column*. It then builds
//! `obsm['spatial']` from `["pxl_row_in_fullres", "pxl_col_in_fullres"]` under
//! its own names, which lands on `(x, y)` after all. Files that keep the
//! renamed columns therefore carry labels that mean the opposite of what 10x
//! means by them, and there is no way to tell from inside the file which
//! convention a given writer used.
//!
//! Checked against 186 Li et al and 50 Vanderbilt files: every one is `(x, y)`
//! when measured against its own shipped image, and every Li et al file
//! disagrees with its own `pxl_*` column names.
//!
//! Everything here is a pure function over slices so it can be tested without
//! an HDF5 file.

/// Fraction of spots that must land on tissue before the mask test says
/// anything. Below this the mask is not describing the section.
const MASK_WINNER_MIN: f64 = 0.70;

/// How far ahead the winning order has to be. Tissue that fills the frame
/// scores high both ways; the margin is what keeps those cases quiet. Over 236
/// real files this left 119 decisive and never produced a contradiction.
const MASK_MARGIN_MIN: f64 = 0.15;

/// Smallest share of the image the mask may cover. Under this it is picking up
/// dust rather than a section.
const MASK_COVERAGE_MIN: f64 = 0.03;

/// Largest share of the image the mask may cover. Over this it is picking up
/// the background.
const MASK_COVERAGE_MAX: f64 = 0.80;

/// How far a pixel has to sit from the background before it counts as tissue.
/// On the 0..1 grey scale of these images. Absolute, so it works on a bright
/// slide and a dark fluorescence frame alike.
const MASK_INTENSITY_DELTA: f64 = 0.10;

/// Which column of `obsm/spatial` holds the `x` coordinate.
///
/// The project contract is `(x, y)`, so [`SpatialOrientation::Yx`] is the one
/// that needs a swap on the way in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SpatialOrientation {
    /// Column 0 is `x`. What `scanpy.read_visium` writes and what all 236
    /// survey files hold, hence the default.
    #[default]
    Xy,
    /// Column 0 is `y`, so the pair needs swapping.
    Yx,
}

impl SpatialOrientation {
    /// Parse the string form used at the R boundary.
    ///
    /// ### Params
    ///
    /// * `s` - `"xy"` or `"yx"`, case insensitive.
    ///
    /// ### Returns
    ///
    /// The matching variant, or `None`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "xy" => Some(Self::Xy),
            "yx" => Some(Self::Yx),
            _ => None,
        }
    }

    /// The string form used at the R boundary.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Xy => "xy",
            Self::Yx => "yx",
        }
    }

    /// Reorder a raw `obsm/spatial` row into the `(x, y)` contract.
    ///
    /// ### Params
    ///
    /// * `c0` - Value in column 0 of `obsm/spatial`.
    /// * `c1` - Value in column 1.
    ///
    /// ### Returns
    ///
    /// The pair as `(x, y)`.
    #[inline]
    pub fn to_xy(&self, c0: f64, c1: f64) -> (f64, f64) {
        match self {
            Self::Xy => (c0, c1),
            Self::Yx => (c1, c0),
        }
    }
}

/// How the orientation was arrived at.
///
/// Ordered by how directly it measures the thing that has a consequence, which
/// is whether the coordinates line up with the histology image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrientationEvidence {
    /// One order puts the spots on tissue in the shipped image and the other
    /// does not. The strongest signal available, because it tests the alignment
    /// itself rather than a proxy for it.
    ImageTissue,
    /// One order keeps every spot inside the frame the shipped image implies
    /// and the other pushes spots off the edge. One-sided: it can refute an
    /// order but never confirm one, so it only settles things when the tissue
    /// mask was unusable.
    ImageFrame,
    /// `obs` carried `pxl_row_in_fullres` and `pxl_col_in_fullres` and one of
    /// them matched a column of `obsm/spatial` element for element.
    ///
    /// Weaker than it looks. `scanpy.read_visium` swaps these two names
    /// relative to Space Ranger, so the labels mean the opposite in a file that
    /// went through it. Only consulted when the file ships no image, where a
    /// transposed read has no observable consequence anyway.
    ObsPixelColumns,
    /// Nothing in the file settled it, so the caller's parameter was taken.
    Assumed,
}

impl OrientationEvidence {
    /// The string form used at the R boundary.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ImageTissue => "image_tissue",
            Self::ImageFrame => "image_frame",
            Self::ObsPixelColumns => "obs_pixel_columns",
            Self::Assumed => "assumed",
        }
    }

    /// Whether this call is worth warning the caller about.
    pub fn is_uncertain(&self) -> bool {
        matches!(self, Self::ObsPixelColumns | Self::Assumed)
    }
}

/// The resolved column order together with how it was resolved.
#[derive(Clone, Copy, Debug)]
pub struct OrientationCall {
    /// The resolved column order.
    pub orientation: SpatialOrientation,
    /// What settled it.
    pub evidence: OrientationEvidence,
}

/// Resolve the column order from the `obs` pixel columns.
///
/// Both columns have to agree on the same assignment before this returns, which
/// rules out the degenerate case where the two happen to be equal.
///
/// Read the module docs before trusting the answer: the names this keys on are
/// swapped by `scanpy.read_visium`, so an exact match says which column of
/// `obsm/spatial` came from which `obs` column and nothing more.
///
/// ### Params
///
/// * `col0` - Column 0 of `obsm/spatial`.
/// * `col1` - Column 1 of `obsm/spatial`.
/// * `pxl_row` - `obs/pxl_row_in_fullres`.
/// * `pxl_col` - `obs/pxl_col_in_fullres`.
///
/// ### Returns
///
/// The orientation the labels imply, or `None` if neither assignment matches
/// exactly.
pub fn orientation_from_obs_pixels(
    col0: &[f64],
    col1: &[f64],
    pxl_row: &[f64],
    pxl_col: &[f64],
) -> Option<SpatialOrientation> {
    let n = col0.len();
    if n == 0 || col1.len() != n || pxl_row.len() != n || pxl_col.len() != n {
        return None;
    }

    let same = |a: &[f64], b: &[f64]| a.iter().zip(b.iter()).all(|(x, y)| x == y);

    if same(col0, pxl_row) && same(col1, pxl_col) {
        return Some(SpatialOrientation::Yx);
    }
    if same(col0, pxl_col) && same(col1, pxl_row) {
        return Some(SpatialOrientation::Xy);
    }

    None
}

/// Resolve the column order by asking which one fits inside the slide.
///
/// The shipped image is the full-res frame scaled by its own scale factor, so
/// dividing its pixel dimensions by that factor recovers the frame the
/// coordinates live in. A spot cannot sit outside the slide it was imaged on,
/// so an order that pushes coordinates past the edge is refuted.
///
/// Only decides anything when exactly one order survives, which was 10 of the
/// 50 Vanderbilt files and 1 of the 186 Li et al ones.
///
/// ### Params
///
/// * `max_col0` - Largest value in column 0 of `obsm/spatial`.
/// * `max_col1` - Largest value in column 1.
/// * `frame_height` - Full-res frame height, i.e. image rows over scale factor.
/// * `frame_width` - Full-res frame width.
///
/// ### Returns
///
/// The orientation when exactly one order fits, otherwise `None`.
pub fn orientation_from_image_frame(
    max_col0: f64,
    max_col1: f64,
    frame_height: f64,
    frame_width: f64,
) -> Option<SpatialOrientation> {
    if !(frame_height.is_finite() && frame_width.is_finite())
        || frame_height <= 0.0
        || frame_width <= 0.0
    {
        return None;
    }

    // Yx: column 0 is the row index, so it is bounded by the frame height.
    let yx_fits = max_col0 <= frame_height && max_col1 <= frame_width;
    let xy_fits = max_col1 <= frame_height && max_col0 <= frame_width;

    match (yx_fits, xy_fits) {
        (true, false) => Some(SpatialOrientation::Yx),
        (false, true) => Some(SpatialOrientation::Xy),
        _ => None,
    }
}

/// A tissue mask over a greyscale image.
///
/// Background is taken from the border ring rather than assumed to be white,
/// so a dark fluorescence frame masks the same way a bright H&E slide does.
///
/// ### Params
///
/// * `grey` - Row-major greyscale image, values on 0..1.
/// * `height` - Rows.
/// * `width` - Columns.
///
/// ### Returns
///
/// The mask, row-major, or `None` when it covers implausibly much or little of
/// the frame.
pub fn tissue_mask(grey: &[f64], height: usize, width: usize) -> Option<Vec<bool>> {
    if height == 0 || width == 0 || grey.len() != height * width {
        return None;
    }

    let mut ring: Vec<f64> = Vec::with_capacity(2 * (height + width));
    for c in 0..width {
        ring.push(grey[c]);
        ring.push(grey[(height - 1) * width + c]);
    }
    for r in 0..height {
        ring.push(grey[r * width]);
        ring.push(grey[r * width + width - 1]);
    }
    ring.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let background = ring[ring.len() / 2];
    if !background.is_finite() {
        return None;
    }

    let mask: Vec<bool> = grey
        .iter()
        .map(|v| (v - background).abs() > MASK_INTENSITY_DELTA)
        .collect();

    let coverage = mask.iter().filter(|m| **m).count() as f64 / mask.len() as f64;
    if !(MASK_COVERAGE_MIN..=MASK_COVERAGE_MAX).contains(&coverage) {
        return None;
    }

    Some(mask)
}

/// Resolve the column order by asking which one puts the spots on the tissue.
///
/// The only test here that measures the consequence rather than a proxy for it:
/// under the right order the spots sit on the section, under the wrong one they
/// sit on its transpose. Deliberately conservative, because a section that
/// fills the frame scores high either way. It stays quiet unless the winner
/// clears [`MASK_WINNER_MIN`] and leads by [`MASK_MARGIN_MIN`].
///
/// ### Params
///
/// * `col0` - Column 0 of `obsm/spatial`, in full-res pixels.
/// * `col1` - Column 1.
/// * `mask` - Row-major tissue mask from [`tissue_mask`].
/// * `height` - Mask rows.
/// * `width` - Mask columns.
/// * `scalef` - Scale from full-res pixels into the mask's pixel space.
///
/// ### Returns
///
/// The orientation when one order is decisively better, otherwise `None`.
pub fn orientation_from_tissue_mask(
    col0: &[f64],
    col1: &[f64],
    mask: &[bool],
    height: usize,
    width: usize,
    scalef: f64,
) -> Option<SpatialOrientation> {
    if col0.is_empty()
        || col0.len() != col1.len()
        || mask.len() != height * width
        || !scalef.is_finite()
        || scalef <= 0.0
    {
        return None;
    }

    let hit_rate = |rows: &[f64], cols: &[f64]| -> f64 {
        let hits = rows
            .iter()
            .zip(cols.iter())
            .filter(|(r, c)| {
                let (r, c) = ((*r * scalef).round(), (*c * scalef).round());
                if r < 0.0 || c < 0.0 || r >= height as f64 || c >= width as f64 {
                    return false;
                }
                mask[r as usize * width + c as usize]
            })
            .count();
        hits as f64 / rows.len() as f64
    };

    // Yx: column 0 indexes rows of the image.
    let yx_score = hit_rate(col0, col1);
    let xy_score = hit_rate(col1, col0);

    let (winner, best, worst) = if yx_score >= xy_score {
        (SpatialOrientation::Yx, yx_score, xy_score)
    } else {
        (SpatialOrientation::Xy, xy_score, yx_score)
    };

    if best < MASK_WINNER_MIN || best - worst < MASK_MARGIN_MIN {
        return None;
    }

    Some(winner)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// A 20 x 40 frame with a solid block of "tissue" in the left third. The
    /// frame is deliberately non-square and the block deliberately off-centre,
    /// so the transpose of a spot cloud lands somewhere else entirely.
    fn wide_mask() -> (Vec<bool>, usize, usize) {
        let (h, w) = (20usize, 40usize);
        let mut mask = vec![false; h * w];
        for r in 4..16 {
            for c in 2..14 {
                mask[r * w + c] = true;
            }
        }
        (mask, h, w)
    }

    #[test]
    fn test_to_xy_swaps_only_for_yx() {
        assert_eq!(SpatialOrientation::Xy.to_xy(3.0, 7.0), (3.0, 7.0));
        assert_eq!(SpatialOrientation::Yx.to_xy(3.0, 7.0), (7.0, 3.0));
    }

    #[test]
    fn test_orientation_round_trips_through_strings() {
        for o in [SpatialOrientation::Xy, SpatialOrientation::Yx] {
            assert_eq!(SpatialOrientation::parse(o.as_str()), Some(o));
        }
        assert_eq!(SpatialOrientation::parse("rowcol"), None);
    }

    #[test]
    fn test_obs_pixels_matches_the_labels_both_ways() {
        let pxl_row = [10.0, 20.0, 30.0];
        let pxl_col = [100.0, 140.0, 180.0];
        assert_eq!(
            orientation_from_obs_pixels(&pxl_row, &pxl_col, &pxl_row, &pxl_col),
            Some(SpatialOrientation::Yx)
        );
        assert_eq!(
            orientation_from_obs_pixels(&pxl_col, &pxl_row, &pxl_row, &pxl_col),
            Some(SpatialOrientation::Xy)
        );
    }

    #[test]
    fn test_obs_pixels_refuses_a_partial_match() {
        let pxl_row = [10.0, 20.0, 30.0];
        let pxl_col = [100.0, 140.0, 180.0];
        let col1 = [1.0, 2.0, 3.0];
        assert_eq!(
            orientation_from_obs_pixels(&pxl_row, &col1, &pxl_row, &pxl_col),
            None
        );
    }

    #[test]
    fn test_obs_pixels_refuses_length_mismatch() {
        let pxl_row = [10.0, 20.0];
        let pxl_col = [100.0, 140.0];
        assert_eq!(
            orientation_from_obs_pixels(&[10.0], &[100.0], &pxl_row, &pxl_col),
            None
        );
    }

    #[test]
    fn test_image_frame_refutes_the_overflowing_order() {
        // From `7319_AS_4_filtered_trimmed.h5ad`: under `Yx` the larger
        // coordinate lands 18% past the bottom of the slide.
        assert_eq!(
            orientation_from_image_frame(9055.0, 6009.0, 7674.0, 11228.0),
            Some(SpatialOrientation::Xy)
        );
    }

    #[test]
    fn test_image_frame_is_silent_when_it_cannot_separate() {
        assert_eq!(
            orientation_from_image_frame(100.0, 100.0, 500.0, 500.0),
            None
        );
        assert_eq!(
            orientation_from_image_frame(900.0, 900.0, 500.0, 500.0),
            None
        );
    }

    #[test]
    fn test_image_frame_rejects_a_degenerate_frame() {
        assert_eq!(orientation_from_image_frame(1.0, 2.0, 0.0, 10.0), None);
        assert_eq!(orientation_from_image_frame(1.0, 2.0, f64::NAN, 10.0), None);
    }

    #[test]
    fn test_tissue_mask_finds_a_dark_section_on_a_bright_slide() {
        let (h, w) = (10usize, 10usize);
        let mut grey = vec![1.0_f64; h * w];
        for r in 3..7 {
            for c in 3..7 {
                grey[r * w + c] = 0.2;
            }
        }
        let mask = tissue_mask(&grey, h, w).unwrap();
        assert_eq!(mask.iter().filter(|m| **m).count(), 16);
    }

    #[test]
    fn test_tissue_mask_finds_a_bright_section_on_a_dark_slide() {
        // Fluorescence: the background is dark and the section is bright. An
        // absolute distance from the border value handles both.
        let (h, w) = (10usize, 10usize);
        let mut grey = vec![0.02_f64; h * w];
        for r in 3..7 {
            for c in 3..7 {
                grey[r * w + c] = 0.9;
            }
        }
        let mask = tissue_mask(&grey, h, w).unwrap();
        assert_eq!(mask.iter().filter(|m| **m).count(), 16);
    }

    #[test]
    fn test_tissue_mask_declines_an_empty_frame() {
        let grey = vec![1.0_f64; 100];
        assert!(tissue_mask(&grey, 10, 10).is_none());
    }

    #[test]
    fn test_tissue_mask_declines_a_frame_that_is_all_tissue() {
        // Nothing but a one-pixel border matches the background, so 87% of the
        // frame masks as tissue and the mask says nothing useful about where
        // the spots are.
        let (h, w) = (30usize, 30usize);
        let mut grey = vec![0.0_f64; h * w];
        for c in 0..w {
            grey[c] = 1.0;
            grey[(h - 1) * w + c] = 1.0;
        }
        for r in 0..h {
            grey[r * w] = 1.0;
            grey[r * w + w - 1] = 1.0;
        }
        assert!(tissue_mask(&grey, h, w).is_none());
    }

    #[test]
    fn test_tissue_mask_picks_the_order_that_lands_on_tissue() {
        let (mask, h, w) = wide_mask();
        // Spots inside rows 5..15 and columns 3..13 of the mask, given at
        // full-res with a scale factor of 0.5.
        let rows: Vec<f64> = (5..15).map(|r| (r as f64) * 2.0).collect();
        let cols: Vec<f64> = (5..15).map(|i| ((i % 10) as f64 + 3.0) * 2.0).collect();

        // Written `Yx`: column 0 holds the row index.
        assert_eq!(
            orientation_from_tissue_mask(&rows, &cols, &mask, h, w, 0.5),
            Some(SpatialOrientation::Yx)
        );
        // The same spots written the other way round have to come back `Xy`,
        // which is the assertion that dies if the branch is inverted.
        assert_eq!(
            orientation_from_tissue_mask(&cols, &rows, &mask, h, w, 0.5),
            Some(SpatialOrientation::Xy)
        );
    }

    #[test]
    fn test_tissue_mask_stays_quiet_without_a_margin() {
        // Tissue on the diagonal is symmetric under a transpose, so both orders
        // score the same and neither is claimed.
        let (h, w) = (20usize, 20usize);
        let mut mask = vec![false; h * w];
        for i in 0..h {
            for d in 0..3 {
                mask[i * w + (i + d).min(w - 1)] = true;
                mask[(i + d).min(h - 1) * w + i] = true;
            }
        }
        let rows: Vec<f64> = (2..18).map(|i| i as f64).collect();
        let cols: Vec<f64> = (2..18).map(|i| i as f64).collect();
        assert_eq!(
            orientation_from_tissue_mask(&rows, &cols, &mask, h, w, 1.0),
            None
        );
    }

    #[test]
    fn test_tissue_mask_stays_quiet_when_nothing_lands_on_tissue() {
        let (mask, h, w) = wide_mask();
        let rows = vec![19.0, 19.0, 19.0];
        let cols = vec![39.0, 38.0, 37.0];
        assert_eq!(
            orientation_from_tissue_mask(&rows, &cols, &mask, h, w, 1.0),
            None
        );
    }

    #[test]
    fn test_swapping_coordinates_preserves_every_distance() {
        // The claim the module docs rest on: a transpose is an isometry, so no
        // distance-based statistic can tell the two apart.
        let pts = [(1.0_f64, 7.0), (3.0, 2.0), (9.0, 4.0), (0.5, 6.5)];
        let swapped: Vec<(f64, f64)> = pts.iter().map(|(x, y)| (*y, *x)).collect();
        for i in 0..pts.len() {
            for j in 0..pts.len() {
                let d = (pts[i].0 - pts[j].0).powi(2) + (pts[i].1 - pts[j].1).powi(2);
                let s =
                    (swapped[i].0 - swapped[j].0).powi(2) + (swapped[i].1 - swapped[j].1).powi(2);
                assert!((d - s).abs() < 1e-12);
            }
        }
    }
}
