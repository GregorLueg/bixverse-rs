//! Per-spot image features against a real Space Ranger output.
//!
//! The dataset is not in the repo: it is a Visium v2 CytAssist run (Space
//! Ranger 2.0.0, 6.5 mm, 4,992 spots) sitting in a local folder. Every test
//! here checks for it and skips loudly if it is absent, so CI stays green
//! without pretending the coverage exists. Point `BIXVERSE_VISIUM_DIR` at a
//! `spatial/` directory to run them.
//!
//! What this does not cover: any vendor slide format. No SVS, NDPI or MIRAX
//! file was available. The synthetic tiled-TIFF tests in `slide.rs` exercise
//! the OpenSlide binding itself, but no real scanner output has ever been
//! through this code.

#![cfg(feature = "spatial-image")]

use std::path::{Path, PathBuf};

use bixverse_rs::spatial::sp_image::source::{AnyImageSource, ImageSource, open_image_source};
use bixverse_rs::spatial::sp_image::{SpatialImageParams, spatial_image_features};

/// Spot diameter in full-res pixels, from `scalefactors_json.json`.
const SPOT_DIAMETER_FULLRES: f32 = 256.123_37;

/// `tissue_hires_scalef` from `scalefactors_json.json`.
const HIRES_SCALEF: f32 = 0.056_121_446;

/// `tissue_lowres_scalef` from `scalefactors_json.json`.
const LOWRES_SCALEF: f32 = 0.016_836_435;

/// Scale from full-res into `cytassist_image.tiff` pixel space.
///
/// Not `regist_target_img_scalef` (0.16836435) as stored. That factor targets
/// the *registration target*, which Space Ranger upsamples 2x from the
/// CytAssist acquisition, so it maps full-res onto a 6000x6000 frame. The file
/// on disk here is 3000x3000, i.e. exactly half, and 35638 * 0.084182 = 3000.
///
/// The consequence for anyone reading the plan: a spot is 21.6 px across in
/// this file, not the 43.1 px quoted for the registration target.
const CYTASSIST_SCALEF: f32 = 0.168_364_35 / 2.0;

/// Spots on a Visium v2 6.5 mm capture area, in tissue or not.
const N_SPOTS_TOTAL: usize = 4992;

/// Spots flagged `in_tissue = 1` in this particular dataset.
const N_SPOTS_IN_TISSUE: usize = 3858;

/// Locate the Visium `spatial/` directory, if it is around.
///
/// ### Returns
///
/// The path, or `None` if neither the environment variable nor the default
/// location resolves.
fn visium_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BIXVERSE_VISIUM_DIR") {
        let p = PathBuf::from(p);
        return p.is_dir().then_some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let p = PathBuf::from(home).join("Desktop/visium_example_data/spatial");
    p.is_dir().then_some(p)
}

/// Read spot centres out of `tissue_positions.csv`.
///
/// Coordinates come back as `(x, y)` full-res pixels, so `pxl_col_in_fullres`
/// first and `pxl_row_in_fullres` second. Getting that pair the wrong way round
/// is the classic Visium bug.
///
/// ### Params
///
/// * `dir` - The `spatial/` directory.
/// * `in_tissue_only` - Keep only spots flagged as under tissue.
///
/// ### Returns
///
/// The coordinates in file order.
fn read_positions(dir: &Path, in_tissue_only: bool) -> Vec<(f32, f32)> {
    let text = std::fs::read_to_string(dir.join("tissue_positions.csv")).unwrap();
    text.lines()
        .skip(1)
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() < 6 {
                return None;
            }
            if in_tissue_only && f[1].trim() != "1" {
                return None;
            }
            let row: f32 = f[4].trim().parse().ok()?;
            let col: f32 = f[5].trim().parse().ok()?;
            Some((col, row))
        })
        .collect()
}

/// Skip helper. Prints why, so a silent pass is never mistaken for coverage.
macro_rules! require_data {
    () => {
        match visium_dir() {
            Some(d) => d,
            None => {
                eprintln!(
                    "SKIPPED: no Visium data found. Set BIXVERSE_VISIUM_DIR to a Space Ranger 'spatial/' directory."
                );
                return;
            }
        }
    };
}

#[test]
fn test_hires_png_yields_a_per_spot_feature_matrix() {
    let dir = require_data!();
    assert_eq!(read_positions(&dir, false).len(), N_SPOTS_TOTAL);
    let coords = read_positions(&dir, true);
    assert_eq!(coords.len(), N_SPOTS_IN_TISSUE);

    let src = open_image_source(dir.join("tissue_hires_image.png"), HIRES_SCALEF, 1.0).unwrap();
    // A PNG has no pyramid, so the probe must fall through to the whole-image
    // backend.
    assert!(matches!(src, AnyImageSource::Whole(_)));
    assert_eq!(src.dims(), (2000, 1650));

    let res = spatial_image_features(&src, &coords, SPOT_DIAMETER_FULLRES, None).unwrap();

    assert_eq!(res.n_spots(), coords.len(), "no spot should fall off");
    assert_eq!(res.n_features(), 29);
    assert_eq!(res.values.len(), res.n_spots() * res.n_features());
    assert!(res.values.iter().all(|v| v.is_finite()));

    // Sanity on the actual numbers rather than just the shape: an H&E section
    // is not uniformly white, and mean OD must sit inside the physical range.
    let names = res.feature_names.clone();
    let od = names.iter().position(|n| n == "mean_od").unwrap();
    let mean_od: f32 = (0..res.n_spots())
        .map(|i| res.row(i).unwrap()[od])
        .sum::<f32>()
        / res.n_spots() as f32;
    assert!(
        mean_od > 0.02 && mean_od < 2.41,
        "mean OD across tissue spots was {mean_od}"
    );
}

#[test]
fn test_lowres_and_hires_agree_on_which_spots_are_darkest() {
    let dir = require_data!();
    let coords = read_positions(&dir, true);

    let hires = open_image_source(dir.join("tissue_hires_image.png"), HIRES_SCALEF, 1.0).unwrap();
    let lowres =
        open_image_source(dir.join("tissue_lowres_image.png"), LOWRES_SCALEF, 1.0).unwrap();
    assert_eq!(lowres.dims(), (600, 495));

    let a = spatial_image_features(&hires, &coords, SPOT_DIAMETER_FULLRES, None).unwrap();
    let b = spatial_image_features(&lowres, &coords, SPOT_DIAMETER_FULLRES, None).unwrap();
    assert_eq!(a.n_spots(), b.n_spots());

    // The two images are the same section at different resolutions, so the
    // per-spot mean optical density has to track. This is the real check that
    // the scale factor is applied correctly: get it wrong and the two sets of
    // tiles land on different tissue and the correlation collapses.
    let od = a.feature_names.iter().position(|n| n == "mean_od").unwrap();
    let xs: Vec<f64> = (0..a.n_spots())
        .map(|i| a.row(i).unwrap()[od] as f64)
        .collect();
    let ys: Vec<f64> = (0..b.n_spots())
        .map(|i| b.row(i).unwrap()[od] as f64)
        .collect();
    let r = pearson(&xs, &ys);
    assert!(r > 0.95, "hires vs lowres mean OD correlation was {r}");
}

#[test]
fn test_cytassist_tiff_is_readable_and_has_the_expected_geometry() {
    let dir = require_data!();
    let path = dir.join("cytassist_image.tiff");

    // This file is a 3000x3000 LZW strip TIFF. OpenSlide's generic-tiff driver
    // needs a *tiled* TIFF, so it declines, and the probe must fall through to
    // the image decoder rather than giving up.
    let src = open_image_source(&path, CYTASSIST_SCALEF, 1.0).unwrap();
    assert!(matches!(src, AnyImageSource::Whole(_)));
    assert_eq!(src.dims(), (3000, 3000));

    let coords = read_positions(&dir, true);
    let res = spatial_image_features(&src, &coords, SPOT_DIAMETER_FULLRES, None).unwrap();
    assert_eq!(res.n_spots(), coords.len());
    assert!(res.values.iter().all(|v| v.is_finite()));

    // 256.12 full-res px * 0.084182 = 21.6 px per spot, against 14.4 in the
    // hires PNG. Still small for a GLCM, but the best this dataset offers.
    let side = (SPOT_DIAMETER_FULLRES * src.scale()).ceil();
    assert!((side - 22.0).abs() < 1.5, "spot side came out at {side}");
}

#[test]
fn test_tile_scale_and_glcm_levels_are_honoured_on_real_data() {
    let dir = require_data!();
    let coords: Vec<(f32, f32)> = read_positions(&dir, true).into_iter().take(200).collect();
    let src = open_image_source(dir.join("cytassist_image.tiff"), CYTASSIST_SCALEF, 1.0).unwrap();

    let coarse = SpatialImageParams {
        glcm_levels: 8,
        ..Default::default()
    };
    let a = spatial_image_features(&src, &coords, SPOT_DIAMETER_FULLRES, None).unwrap();
    let b = spatial_image_features(&src, &coords, SPOT_DIAMETER_FULLRES, Some(coarse)).unwrap();

    let at = a
        .feature_names
        .iter()
        .position(|n| n == "glcm_grey_entropy")
        .unwrap();
    // Fewer grey levels cannot produce more entropy.
    for i in 0..a.n_spots() {
        assert!(b.row(i).unwrap()[at] <= a.row(i).unwrap()[at] + 1e-4);
    }
}

/// Pearson correlation.
///
/// ### Params
///
/// * `x` - First vector.
/// * `y` - Second vector, same length.
///
/// ### Returns
///
/// The coefficient, or 0.0 if either vector is constant.
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    for (a, b) in x.iter().zip(y.iter()) {
        let u = a - mx;
        let v = b - my;
        num += u * v;
        dx += u * u;
        dy += v * v;
    }
    if dx <= 0.0 || dy <= 0.0 {
        return 0.0;
    }
    num / (dx * dy).sqrt()
}
