//! Histology image features per spot.
//!
//! Turns whatever image came with a spatial run into a per-spot feature matrix
//! with the same shape as any other per-spot assay. It joins to the
//! transcriptional data on spot index and needs no further machinery.
//!
//! ## Index space contract
//!
//! Coordinates handed to this module are **full-res pixels**, `f32`, `(x, y)`,
//! always, whichever image is loaded. The image layer applies the scale factor
//! and nothing else does. Spot indices in [`SpatialImageRes`] are 0-based and
//! positional against the coordinate slice that went in.
//!
//! ## Layout
//!
//! * `source.rs` holds the [`ImageSource`] trait, the whole-image
//!   implementation and the probe that picks between backends.
//! * `slide.rs` reads windows out of an OpenSlide pyramid.
//! * `tiles.rs` cuts per-spot windows.
//! * `colour.rs` does Ruifrok-Johnston colour deconvolution.
//! * `texture.rs` does GLCM and the Haralick reductions.
//!
//! ## Coverage warning
//!
//! The slide path in `slide.rs` has never run against a real scanner file. No
//! `.svs`, `.ndpi` or MIRAX image was available, and the CytAssist TIFF that
//! ships with the Space Ranger example is a plain strip TIFF that OpenSlide
//! declines to open. The whole-image path is exercised against real Space
//! Ranger PNGs; the slide path is checked against its API contract and
//! synthetic fixtures only.
//!
//! ## Licensing
//!
//! OpenSlide is LGPL-2.1 and this crate is MIT, which works only as long as the
//! C library is linked dynamically. `Cargo.toml` forces the `dynamic-link`
//! feature on `openslide-sys` for exactly that reason. Do not link it
//! statically and do not swap in a pure-Rust LGPL-only reader.
//!
//! ### References
//!
//! Ruifrok & Johnston, Anal Quant Cytol Histol, 2001 (colour deconvolution)
//!
//! Haralick, Shanmugam & Dinstein, IEEE Trans Syst Man Cybern, 1973 (texture)

pub mod colour;
pub mod slide;
pub mod source;
pub mod texture;
pub mod tiles;

use rayon::prelude::*;

use crate::prelude::*;
use crate::spatial::sp_image::colour::{StainMatrix, mean_optical_density};
use crate::spatial::sp_image::source::{ImageSource, RgbTile};
use crate::spatial::sp_image::texture::{
    DEFAULT_GLCM_LEVELS, DEFAULT_GLCM_OFFSETS, GlcmScratch, HARALICK_FEATURE_NAMES,
    N_HARALICK_FEATURES, STAIN_QUANT_MAX, glcm_haralick, quantise_stain, quantise_u8,
};
use crate::spatial::sp_image::tiles::spot_window;

///////////////
// Constants //
///////////////

/// Channels that get first-order statistics, in output order.
const CHANNEL_NAMES: [&str; 6] = ["r", "g", "b", "grey", "haem", "eosin"];

/// First-order statistics per channel: mean, SD, entropy.
const N_FIRST_ORDER: usize = 3;

/// Channels that get a GLCM, in output order.
///
/// Grey and haematoxylin. Grey is the conventional choice; haematoxylin is
/// there because for anything nuclear it carries the density signal that grey
/// averages away against eosin.
const GLCM_CHANNEL_NAMES: [&str; 2] = ["grey", "haem"];

/// Total features per spot.
const N_SPOT_FEATURES: usize =
    CHANNEL_NAMES.len() * N_FIRST_ORDER + 1 + GLCM_CHANNEL_NAMES.len() * N_HARALICK_FEATURES;

/// Histogram bins for the first-order Shannon entropy.
///
/// 8-bit channels get one bin per byte. Stain concentrations get the same
/// count spread over `[0, STAIN_QUANT_MAX]`, so entropies stay comparable
/// between the two kinds of channel.
const ENTROPY_BINS: usize = 256;

/// BT.601 luma weights for the RGB to grey conversion.
const LUMA_WEIGHTS: [f32; 3] = [0.299, 0.587, 0.114];

////////////
// Params //
////////////

/// Knobs for the per-spot image feature run.
#[derive(Clone, Debug)]
pub struct SpatialImageParams {
    /// Multiplier on the spot diameter when cutting a tile. 1.0 takes the spot
    /// itself, larger values pull in surrounding context.
    pub tile_scale: f32,
    /// Grey levels the GLCM quantises to. More levels resolve more texture but
    /// need `levels^2` worth of pixels to fill the matrix.
    pub glcm_levels: u8,
    /// Neighbour offsets as `(dy, dx)`, all accumulated into one matrix.
    pub glcm_offsets: Vec<(i8, i8)>,
    /// Stain basis for colour deconvolution.
    pub stain_matrix: StainMatrix,
}

impl SpatialImageParams {
    /// Build the parameters.
    ///
    /// ### Params
    ///
    /// * `tile_scale` - Multiplier on the spot diameter.
    /// * `glcm_levels` - Grey levels for quantisation, 2 to 255.
    /// * `glcm_offsets` - Neighbour offsets as `(dy, dx)`.
    /// * `stain_matrix` - Stain basis.
    ///
    /// ### Returns
    ///
    /// The parameters.
    pub fn new(
        tile_scale: f32,
        glcm_levels: u8,
        glcm_offsets: Vec<(i8, i8)>,
        stain_matrix: StainMatrix,
    ) -> Self {
        Self {
            tile_scale,
            glcm_levels,
            glcm_offsets,
            stain_matrix,
        }
    }
}

impl Default for SpatialImageParams {
    /// Tile at the spot diameter, 32 grey levels, four offsets at distance 1
    /// and the Ruifrok-Johnston H&E basis.
    fn default() -> Self {
        Self {
            tile_scale: 1.0,
            glcm_levels: DEFAULT_GLCM_LEVELS,
            glcm_offsets: DEFAULT_GLCM_OFFSETS.to_vec(),
            stain_matrix: StainMatrix::default(),
        }
    }
}

/////////////
// Results //
/////////////

/// Per-spot image features.
#[derive(Clone, Debug)]
pub struct SpatialImageRes {
    /// 0-based positional indices into the coordinate slice that went in. Spots
    /// whose tile fell entirely off the image are absent, so this can be
    /// shorter than the input.
    pub spot_indices: Vec<usize>,
    /// Feature names, one per column.
    pub feature_names: Vec<String>,
    /// `n_spots * n_features`, row-major, spot-major.
    pub values: Vec<f32>,
}

impl SpatialImageRes {
    /// Number of spots that produced features.
    ///
    /// ### Returns
    ///
    /// The row count.
    pub fn n_spots(&self) -> usize {
        self.spot_indices.len()
    }

    /// Number of features per spot.
    ///
    /// ### Returns
    ///
    /// The column count.
    pub fn n_features(&self) -> usize {
        self.feature_names.len()
    }

    /// The feature row for one result row.
    ///
    /// ### Params
    ///
    /// * `row` - Row index, positional against `spot_indices`.
    ///
    /// ### Returns
    ///
    /// The row, or `None` if out of range.
    pub fn row(&self, row: usize) -> Option<&[f32]> {
        let n = self.n_features();
        self.values.get(row * n..(row + 1) * n)
    }
}

/// Feature names in output order.
///
/// ### Returns
///
/// `n_features` names: first-order stats per channel, then mean optical
/// density, then the Haralick features per GLCM channel.
pub fn feature_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_SPOT_FEATURES);
    for ch in CHANNEL_NAMES {
        names.push(format!("{ch}_mean"));
        names.push(format!("{ch}_sd"));
        names.push(format!("{ch}_entropy"));
    }
    names.push("mean_od".to_string());
    for ch in GLCM_CHANNEL_NAMES {
        for h in HARALICK_FEATURE_NAMES {
            names.push(format!("glcm_{ch}_{h}"));
        }
    }
    names
}

/////////////
// Scratch //
/////////////

/// Per-thread buffers for one spot's worth of work.
///
/// Everything here is cleared and refilled per spot, so a thread allocates once
/// and then stops. The tiles themselves are the big items and never outlive the
/// spot that produced them, which is what keeps a 4,992-spot full-res run off
/// the ~1 GB a materialised tile set would cost.
#[derive(Debug, Default)]
struct SpotScratch {
    /// Haematoxylin concentrations, one per pixel.
    haem: Vec<f32>,
    /// Eosin concentrations, one per pixel.
    eosin: Vec<f32>,
    /// Residual concentrations, one per pixel. Computed but not featurised.
    residual: Vec<f32>,
    /// Grey channel as bytes, one per pixel.
    grey: Vec<u8>,
    /// GLCM counting buffers.
    glcm: GlcmScratch,
    /// Histogram for the first-order entropy.
    hist: Vec<f64>,
}

impl SpotScratch {
    /// Fresh scratch with the histogram sized and everything else empty.
    fn new() -> Self {
        Self {
            hist: vec![0.0; ENTROPY_BINS],
            ..Default::default()
        }
    }
}

/////////////////////
// First-order set //
/////////////////////

/// Mean, SD and Shannon entropy of an 8-bit channel.
///
/// Accumulated in `f64`. A 256x256 tile is 65k pixels and the naive sum of
/// squares on `f32` loses precision well before that.
///
/// ### Params
///
/// * `values` - The channel.
/// * `hist` - Reusable histogram of at least [`ENTROPY_BINS`] entries.
///
/// ### Returns
///
/// `[mean, sd, entropy]`. All zero for an empty channel. SD is the population
/// SD, dividing by `n`. Entropy is in bits over 256 bins.
fn first_order_u8(values: &[u8], hist: &mut [f64]) -> [f32; N_FIRST_ORDER] {
    if values.is_empty() {
        return [0.0; N_FIRST_ORDER];
    }
    hist[..ENTROPY_BINS].fill(0.0);

    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for &v in values {
        let x = f64::from(v);
        sum += x;
        sum_sq += x * x;
        hist[v as usize] += 1.0;
    }

    let n = values.len() as f64;
    let mean = sum / n;
    let sd = (sum_sq / n - mean * mean).max(0.0).sqrt();
    [
        mean as f32,
        sd as f32,
        shannon_entropy(&hist[..ENTROPY_BINS], n),
    ]
}

/// Mean, SD and Shannon entropy of a stain concentration channel.
///
/// The entropy bins span `[0, STAIN_QUANT_MAX]` rather than the channel's own
/// range, so the value is comparable between spots. Mean and SD use the raw
/// concentrations and are not binned.
///
/// ### Params
///
/// * `values` - Concentrations in optical density.
/// * `hist` - Reusable histogram of at least [`ENTROPY_BINS`] entries.
///
/// ### Returns
///
/// `[mean, sd, entropy]`, all zero for an empty channel.
fn first_order_stain(values: &[f32], hist: &mut [f64]) -> [f32; N_FIRST_ORDER] {
    if values.is_empty() {
        return [0.0; N_FIRST_ORDER];
    }
    hist[..ENTROPY_BINS].fill(0.0);

    let top = (ENTROPY_BINS - 1) as f32;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for &v in values {
        let x = f64::from(v);
        sum += x;
        sum_sq += x * x;
        let bin = (v / STAIN_QUANT_MAX * ENTROPY_BINS as f32).floor();
        let bin = if bin <= 0.0 {
            0
        } else if bin >= top {
            ENTROPY_BINS - 1
        } else {
            bin as usize
        };
        hist[bin] += 1.0;
    }

    let n = values.len() as f64;
    let mean = sum / n;
    let sd = (sum_sq / n - mean * mean).max(0.0).sqrt();
    [
        mean as f32,
        sd as f32,
        shannon_entropy(&hist[..ENTROPY_BINS], n),
    ]
}

/// Shannon entropy of a histogram, in bits.
///
/// ### Params
///
/// * `hist` - Bin counts.
/// * `total` - Sum of `hist`, strictly positive.
///
/// ### Returns
///
/// The entropy.
fn shannon_entropy(hist: &[f64], total: f64) -> f32 {
    let inv = 1.0 / total;
    let mut acc = 0.0_f64;
    for &c in hist {
        if c > 0.0 {
            let p = c * inv;
            acc -= p * p.log2();
        }
    }
    acc as f32
}

/// Convert a tile to a grey channel with BT.601 luma weights.
///
/// ### Params
///
/// * `tile` - Source tile.
/// * `out` - Output buffer, cleared and refilled.
fn to_grey(tile: &RgbTile, out: &mut Vec<u8>) {
    out.clear();
    out.reserve(tile.n_pixels());
    out.extend(tile.pixels().map(|px| {
        let g = LUMA_WEIGHTS[0] * f32::from(px[0])
            + LUMA_WEIGHTS[1] * f32::from(px[1])
            + LUMA_WEIGHTS[2] * f32::from(px[2]);
        g.round().clamp(0.0, 255.0) as u8
    }));
}

////////////
// Driver //
////////////

/// Compute the full feature row for one tile.
///
/// ### Params
///
/// * `tile` - The spot's tile.
/// * `params` - Run parameters.
/// * `scratch` - Reusable per-thread buffers.
///
/// ### Returns
///
/// The feature row in [`feature_names`] order.
fn tile_features(
    tile: &RgbTile,
    params: &SpatialImageParams,
    scratch: &mut SpotScratch,
) -> Result<[f32; N_SPOT_FEATURES], BixverseErrors> {
    let mut out = [0.0_f32; N_SPOT_FEATURES];
    let mut at = 0;

    // R, G, B. Deinterleaved into the grey buffer in turn to avoid a fourth
    // scratch vector; grey is rebuilt properly straight afterwards.
    for offset in 0..3 {
        scratch.grey.clear();
        scratch.grey.extend(tile.pixels().map(|px| px[offset]));
        out[at..at + N_FIRST_ORDER]
            .copy_from_slice(&first_order_u8(&scratch.grey, &mut scratch.hist));
        at += N_FIRST_ORDER;
    }

    to_grey(tile, &mut scratch.grey);
    out[at..at + N_FIRST_ORDER].copy_from_slice(&first_order_u8(&scratch.grey, &mut scratch.hist));
    at += N_FIRST_ORDER;

    params.stain_matrix.deconvolve_tile(
        tile,
        &mut scratch.haem,
        &mut scratch.eosin,
        &mut scratch.residual,
    );
    out[at..at + N_FIRST_ORDER]
        .copy_from_slice(&first_order_stain(&scratch.haem, &mut scratch.hist));
    at += N_FIRST_ORDER;
    out[at..at + N_FIRST_ORDER]
        .copy_from_slice(&first_order_stain(&scratch.eosin, &mut scratch.hist));
    at += N_FIRST_ORDER;

    out[at] = mean_optical_density(tile);
    at += 1;

    // GLCM on grey, which `scratch.grey` still holds.
    quantise_u8(
        &scratch.grey,
        params.glcm_levels,
        &mut scratch.glcm.quantised,
    );
    let grey_glcm = glcm_on_scratch(tile, params, scratch)?;
    out[at..at + N_HARALICK_FEATURES].copy_from_slice(&grey_glcm.to_array());
    at += N_HARALICK_FEATURES;

    // GLCM on haematoxylin.
    quantise_stain(
        &scratch.haem,
        params.glcm_levels,
        &mut scratch.glcm.quantised,
    );
    let haem_glcm = glcm_on_scratch(tile, params, scratch)?;
    out[at..at + N_HARALICK_FEATURES].copy_from_slice(&haem_glcm.to_array());

    Ok(out)
}

/// Run the GLCM over whatever is currently in `scratch.glcm.quantised`.
///
/// ### Params
///
/// * `tile` - The tile, for its dimensions.
/// * `params` - Run parameters.
/// * `scratch` - Buffers, with `glcm.quantised` already filled.
///
/// ### Returns
///
/// The Haralick features.
fn glcm_on_scratch(
    tile: &RgbTile,
    params: &SpatialImageParams,
    scratch: &mut SpotScratch,
) -> Result<texture::HaralickFeatures, BixverseErrors> {
    // Split the borrow: the quantised buffer is read while the counts are
    // written, and both live in the same struct.
    let quantised = std::mem::take(&mut scratch.glcm.quantised);
    let res = glcm_haralick(
        &quantised,
        tile.width,
        tile.height,
        params.glcm_levels,
        &params.glcm_offsets,
        &mut scratch.glcm,
    );
    scratch.glcm.quantised = quantised;
    res
}

/// Per-spot image features off any [`ImageSource`].
///
/// Reading and reduction are fused: a tile is cut, reduced to
/// [`N_SPOT_FEATURES`] numbers and dropped before the next one, so peak memory
/// is one tile per thread rather than one per spot. Parallel over spots with a
/// per-thread scratch set.
///
/// Spots whose tile falls entirely outside the image are dropped rather than
/// failing the run, since a coordinate off the edge of a cropped image is a
/// normal thing rather than a bug. Which spots survived is in
/// [`SpatialImageRes::spot_indices`].
///
/// ### Params
///
/// * `source` - Image to read from.
/// * `coords_fullres` - Spot centres as `(x, y)` in full-res pixels.
/// * `spot_diameter_fullres` - Spot diameter in full-res pixels, i.e.
///   `spot_diameter_fullres` from `scalefactors_json.json`.
/// * `params` - Run parameters, [`SpatialImageParams::default`] if `None`.
///
/// ### Returns
///
/// The per-spot feature matrix, `n_spots * n_features`, row-major and
/// spot-major.
///
/// ### References
///
/// Ruifrok & Johnston, Anal Quant Cytol Histol, 2001
///
/// Haralick, Shanmugam & Dinstein, IEEE Trans Syst Man Cybern, 1973
pub fn spatial_image_features<S: ImageSource>(
    source: &S,
    coords_fullres: &[(f32, f32)],
    spot_diameter_fullres: f32,
    params: Option<SpatialImageParams>,
) -> Result<SpatialImageRes, BixverseErrors> {
    let params = params.unwrap_or_default();
    if params.glcm_levels < 2 {
        return Err(BixverseErrors::GlcmLevelsOutOfRange(params.glcm_levels));
    }
    if params.glcm_offsets.is_empty() {
        return Err(BixverseErrors::GlcmNoOffsets);
    }

    // `map_init` rather than `for_each_init`: same per-thread scratch, but the
    // collect stays ordered so no sort is needed afterwards.
    let rows: Vec<Option<[f32; N_SPOT_FEATURES]>> = coords_fullres
        .par_iter()
        .map_init(SpotScratch::new, |scratch, &coord| {
            let window = match spot_window(source, coord, spot_diameter_fullres, params.tile_scale)
            {
                Ok(w) => w,
                Err(BixverseErrors::ImageWindowOutOfBounds { .. }) => return Ok(None),
                Err(e) => return Err(e),
            };
            let tile = source.read_window(window)?;
            Ok(Some(tile_features(&tile, &params, scratch)?))
        })
        .collect::<Result<Vec<_>, BixverseErrors>>()?;

    let mut spot_indices = Vec::with_capacity(rows.len());
    let mut values = Vec::with_capacity(rows.len() * N_SPOT_FEATURES);
    for (idx, row) in rows.into_iter().enumerate() {
        if let Some(row) = row {
            spot_indices.push(idx);
            values.extend_from_slice(&row);
        }
    }

    Ok(SpatialImageRes {
        spot_indices,
        feature_names: feature_names(),
        values,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::sp_image::source::WholeImage;
    use approx::assert_relative_eq;

    /// A 64x64 image split into a dark left half and a light right half.
    fn two_tone() -> WholeImage {
        let mut data = Vec::with_capacity(64 * 64 * 3);
        for _y in 0..64 {
            for x in 0..64 {
                if x < 32 {
                    data.extend_from_slice(&[60, 40, 90]);
                } else {
                    data.extend_from_slice(&[220, 200, 230]);
                }
            }
        }
        WholeImage::from_rgb8(data, 64, 64, 1.0)
    }

    #[test]
    fn test_feature_names_match_the_row_width() {
        let names = feature_names();
        assert_eq!(names.len(), N_SPOT_FEATURES);
        assert_eq!(names.len(), 29);
        assert_eq!(names[0], "r_mean");
        assert_eq!(names[17], "eosin_entropy");
        assert_eq!(names[18], "mean_od");
        assert_eq!(names[19], "glcm_grey_contrast");
        assert_eq!(names[28], "glcm_haem_entropy");
    }

    #[test]
    fn test_feature_names_are_unique() {
        let names = feature_names();
        let uniq: std::collections::BTreeSet<_> = names.iter().collect();
        assert_eq!(uniq.len(), names.len());
    }

    #[test]
    fn test_features_shape_and_finiteness() {
        let img = two_tone();
        let coords = [(16.0, 16.0), (48.0, 48.0), (32.0, 32.0)];
        let res = spatial_image_features(&img, &coords, 16.0, None).unwrap();

        assert_eq!(res.n_spots(), 3);
        assert_eq!(res.n_features(), N_SPOT_FEATURES);
        assert_eq!(res.values.len(), 3 * N_SPOT_FEATURES);
        assert_eq!(res.spot_indices, vec![0, 1, 2]);
        assert!(res.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_uniform_tiles_recover_their_own_colour() {
        let img = two_tone();
        // A tile wholly inside the dark half.
        let res = spatial_image_features(&img, &[(16.0, 32.0)], 16.0, None).unwrap();
        let row = res.row(0).unwrap();
        let names = feature_names();
        let at = |n: &str| names.iter().position(|x| x == n).unwrap();

        assert_relative_eq!(row[at("r_mean")], 60.0, epsilon = 1e-3);
        assert_relative_eq!(row[at("g_mean")], 40.0, epsilon = 1e-3);
        assert_relative_eq!(row[at("b_mean")], 90.0, epsilon = 1e-3);
        // Uniform tile, so no spread and no texture.
        assert_relative_eq!(row[at("r_sd")], 0.0, epsilon = 1e-6);
        assert_relative_eq!(row[at("r_entropy")], 0.0, epsilon = 1e-6);
        assert_relative_eq!(row[at("glcm_grey_contrast")], 0.0, epsilon = 1e-6);
        assert_relative_eq!(row[at("glcm_grey_energy")], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_darker_tile_has_higher_optical_density() {
        let img = two_tone();
        let res = spatial_image_features(&img, &[(16.0, 32.0), (48.0, 32.0)], 16.0, None).unwrap();
        let names = feature_names();
        let at = names.iter().position(|x| x == "mean_od").unwrap();
        assert!(res.row(0).unwrap()[at] > res.row(1).unwrap()[at]);
    }

    #[test]
    fn test_tile_straddling_the_edge_has_texture_the_uniform_one_lacks() {
        let img = two_tone();
        let res = spatial_image_features(&img, &[(32.0, 32.0), (16.0, 32.0)], 16.0, None).unwrap();
        let names = feature_names();
        let at = names
            .iter()
            .position(|x| x == "glcm_grey_contrast")
            .unwrap();
        let straddle = res.row(0).unwrap()[at];
        let uniform = res.row(1).unwrap()[at];
        assert!(straddle > uniform, "{straddle} should exceed {uniform}");
        assert_relative_eq!(uniform, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_off_image_spots_are_dropped_not_fatal() {
        let img = two_tone();
        let coords = [(16.0, 16.0), (5000.0, 5000.0), (48.0, 48.0)];
        let res = spatial_image_features(&img, &coords, 16.0, None).unwrap();
        assert_eq!(res.n_spots(), 2);
        assert_eq!(res.spot_indices, vec![0, 2]);
    }

    #[test]
    fn test_empty_coords_gives_empty_result() {
        let img = two_tone();
        let res = spatial_image_features(&img, &[], 16.0, None).unwrap();
        assert_eq!(res.n_spots(), 0);
        assert!(res.values.is_empty());
        assert_eq!(res.n_features(), N_SPOT_FEATURES);
    }

    #[test]
    fn test_results_are_deterministic_across_runs() {
        let img = two_tone();
        let coords: Vec<(f32, f32)> = (8..56).map(|i| (i as f32, i as f32)).collect();
        let a = spatial_image_features(&img, &coords, 8.0, None).unwrap();
        let b = spatial_image_features(&img, &coords, 8.0, None).unwrap();
        assert_eq!(a.values, b.values);
        assert_eq!(a.spot_indices, b.spot_indices);
    }

    #[test]
    fn test_scale_factor_shifts_which_pixels_are_read() {
        // Same coordinates, two images differing only in scale, must disagree.
        let mut data = Vec::with_capacity(64 * 64 * 3);
        for y in 0..64u8 {
            for x in 0..64u8 {
                data.extend_from_slice(&[x.wrapping_mul(4), y.wrapping_mul(4), 0]);
            }
        }
        let full = WholeImage::from_rgb8(data.clone(), 64, 64, 1.0);
        let half = WholeImage::from_rgb8(data, 64, 64, 0.5);

        let a = spatial_image_features(&full, &[(40.0, 40.0)], 8.0, None).unwrap();
        let b = spatial_image_features(&half, &[(40.0, 40.0)], 8.0, None).unwrap();
        let names = feature_names();
        let at = names.iter().position(|x| x == "r_mean").unwrap();
        assert!(a.row(0).unwrap()[at] > b.row(0).unwrap()[at]);
    }

    #[test]
    fn test_bad_params_are_rejected() {
        let img = two_tone();
        let p = SpatialImageParams {
            glcm_levels: 1,
            ..Default::default()
        };
        assert!(matches!(
            spatial_image_features(&img, &[(16.0, 16.0)], 16.0, Some(p)),
            Err(BixverseErrors::GlcmLevelsOutOfRange(1))
        ));

        let p = SpatialImageParams {
            glcm_offsets: Vec::new(),
            ..Default::default()
        };
        assert!(matches!(
            spatial_image_features(&img, &[(16.0, 16.0)], 16.0, Some(p)),
            Err(BixverseErrors::GlcmNoOffsets)
        ));
    }

    #[test]
    fn test_tile_scale_changes_the_features() {
        let img = two_tone();
        let wide = SpatialImageParams {
            tile_scale: 2.0,
            ..Default::default()
        };

        // Narrow window is x in [24, 32), wholly inside the dark half. Doubling
        // it gives x in [20, 36), which crosses the boundary at 32.
        let narrow = spatial_image_features(&img, &[(28.0, 32.0)], 8.0, None).unwrap();
        let broad = spatial_image_features(&img, &[(28.0, 32.0)], 8.0, Some(wide)).unwrap();
        let names = feature_names();
        let at = names.iter().position(|x| x == "grey_sd").unwrap();
        // The narrow tile sits inside the dark half, the wide one reaches the
        // boundary.
        assert_relative_eq!(narrow.row(0).unwrap()[at], 0.0, epsilon = 1e-6);
        assert!(broad.row(0).unwrap()[at] > 0.0);
    }

    #[test]
    fn test_first_order_u8_against_hand_values() {
        let mut hist = vec![0.0; ENTROPY_BINS];
        // Two levels, equal counts: mean 5, SD 5, entropy 1 bit.
        let v = [0u8, 10, 0, 10];
        let [mean, sd, ent] = first_order_u8(&v, &mut hist);
        assert_relative_eq!(mean, 5.0, epsilon = 1e-6);
        assert_relative_eq!(sd, 5.0, epsilon = 1e-6);
        assert_relative_eq!(ent, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_first_order_u8_constant_channel() {
        let mut hist = vec![0.0; ENTROPY_BINS];
        let [mean, sd, ent] = first_order_u8(&[7u8; 16], &mut hist);
        assert_relative_eq!(mean, 7.0, epsilon = 1e-6);
        assert_relative_eq!(sd, 0.0, epsilon = 1e-6);
        assert_relative_eq!(ent, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_first_order_stain_bins_are_fixed_not_data_driven() {
        let mut hist = vec![0.0; ENTROPY_BINS];
        // Same shape, different location: entropy must differ because the bins
        // do not follow the data.
        let a = first_order_stain(&[0.0, 0.01], &mut hist);
        let b = first_order_stain(&[0.0, 2.0], &mut hist);
        assert_relative_eq!(a[2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(b[2], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_to_grey_luma_weights() {
        let tile = RgbTile {
            data: vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255],
            width: 4,
            height: 1,
        };
        let mut grey = Vec::new();
        to_grey(&tile, &mut grey);
        assert_eq!(grey, vec![76, 150, 29, 255]);
    }

    #[test]
    fn test_extract_tiles_and_fused_driver_see_the_same_pixels() {
        use crate::spatial::sp_image::tiles::extract_spot_tiles;

        let img = two_tone();
        let coords = [(16.0, 16.0), (48.0, 48.0)];
        let tiles = extract_spot_tiles(&img, &coords, 16.0, 1.0).unwrap();
        let res = spatial_image_features(&img, &coords, 16.0, None).unwrap();

        let params = SpatialImageParams::default();
        let mut scratch = SpotScratch::new();
        for (i, tile) in tiles.iter().enumerate() {
            let want = tile_features(tile, &params, &mut scratch).unwrap();
            assert_eq!(res.row(i).unwrap(), &want[..]);
        }
    }
}
