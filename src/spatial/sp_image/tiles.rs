//! Cutting per-spot tiles out of whatever image is loaded.
//!
//! Coordinates arriving here are full-res pixels. The scale factor is applied
//! exactly once, right here, using [`ImageSource::scale`]. Nothing above this
//! file knows or cares which image is underneath.

use rayon::prelude::*;

use crate::prelude::*;
use crate::spatial::sp_image::source::{ImageSource, ImageWindow, RgbTile};

/// Window for one spot, in the source image's pixel space.
///
/// Side length is `spot_diameter_fullres * tile_scale * source.scale()`,
/// rounded up, minimum 1 pixel. The window is centred on the spot and clipped
/// to the image, so spots near an edge come back with a smaller tile rather
/// than a shifted one. Shifting would move the tile off the spot, which matters
/// more than the tiles all being the same size.
///
/// ### Params
///
/// * `source` - The image the window indexes into.
/// * `coord_fullres` - Spot centre as `(x, y)` in full-res pixels.
/// * `spot_diameter_fullres` - Spot diameter in full-res pixels, e.g.
///   `spot_diameter_fullres` out of `scalefactors_json.json`.
/// * `tile_scale` - Multiplier on the diameter. 1.0 takes the spot, 2.0 takes
///   the spot plus a ring of context around it.
///
/// ### Returns
///
/// The clipped window, or [`BixverseErrors::ImageWindowOutOfBounds`] if the
/// spot falls entirely off the image.
pub fn spot_window<S: ImageSource>(
    source: &S,
    coord_fullres: (f32, f32),
    spot_diameter_fullres: f32,
    tile_scale: f32,
) -> Result<ImageWindow, BixverseErrors> {
    let (img_w, img_h) = source.dims();
    let scale = source.scale();

    let side = (spot_diameter_fullres * tile_scale * scale).ceil().max(1.0) as i64;
    let cx = f64::from(coord_fullres.0 * scale);
    let cy = f64::from(coord_fullres.1 * scale);

    // Work in i64 so a spot hanging off the top-left does not wrap a u32.
    let x0 = (cx - side as f64 / 2.0).round() as i64;
    let y0 = (cy - side as f64 / 2.0).round() as i64;

    let xs = x0.max(0);
    let ys = y0.max(0);
    let xe = (x0 + side).min(i64::from(img_w));
    let ye = (y0 + side).min(i64::from(img_h));

    if xe <= xs || ye <= ys {
        return Err(BixverseErrors::ImageWindowOutOfBounds {
            x: x0.max(0) as u32,
            y: y0.max(0) as u32,
            width: side as u32,
            height: side as u32,
            img_width: img_w,
            img_height: img_h,
        });
    }

    Ok(ImageWindow::new(
        xs as u32,
        ys as u32,
        (xe - xs) as u32,
        (ye - ys) as u32,
    ))
}

/// Cut a tile per spot.
///
/// Parallel over spots. Every tile is materialised, so this is the convenient
/// path rather than the cheap one: 4,992 Visium spots at full-res 256 px is
/// roughly 980 MB of RGB. Use
/// [`spatial_image_features`](crate::spatial::sp_image::spatial_image_features)
/// when only the features are wanted, since it fuses the read and the reduction
/// and never holds more than one tile per thread.
///
/// ### Params
///
/// * `source` - The image to read from.
/// * `coords_fullres` - Spot centres as `(x, y)` in full-res pixels.
/// * `spot_diameter_fullres` - Spot diameter in full-res pixels.
/// * `tile_scale` - Multiplier on the diameter.
///
/// ### Returns
///
/// One tile per spot, in input order. Errors if any spot falls off the image;
/// use [`spot_window`] directly to filter those first.
pub fn extract_spot_tiles<S: ImageSource>(
    source: &S,
    coords_fullres: &[(f32, f32)],
    spot_diameter_fullres: f32,
    tile_scale: f32,
) -> Result<Vec<RgbTile>, BixverseErrors> {
    coords_fullres
        .par_iter()
        .map(|&c| {
            let w = spot_window(source, c, spot_diameter_fullres, tile_scale)?;
            source.read_window(w)
        })
        .collect()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::sp_image::source::WholeImage;

    /// A 100x100 image where pixel `(x, y)` is `[x, y, 0]`.
    fn grid(scale: f32) -> WholeImage {
        let mut data = Vec::with_capacity(100 * 100 * 3);
        for y in 0..100u8 {
            for x in 0..100u8 {
                data.extend_from_slice(&[x, y, 0]);
            }
        }
        WholeImage::from_rgb8(data, 100, 100, scale)
    }

    #[test]
    fn test_spot_window_centres_on_the_spot() {
        let img = grid(1.0);
        let w = spot_window(&img, (50.0, 50.0), 10.0, 1.0).unwrap();
        assert_eq!(w, ImageWindow::new(45, 45, 10, 10));
    }

    #[test]
    fn test_spot_window_applies_the_scale_factor() {
        // Half-scale image: a full-res coordinate of 50 lands at pixel 25 and
        // a 10 px full-res spot becomes 5 px.
        let img = grid(0.5);
        let w = spot_window(&img, (50.0, 50.0), 10.0, 1.0).unwrap();
        assert_eq!(w, ImageWindow::new(23, 23, 5, 5));
    }

    #[test]
    fn test_spot_window_tile_scale_widens_the_window() {
        let img = grid(1.0);
        let w = spot_window(&img, (50.0, 50.0), 10.0, 2.0).unwrap();
        assert_eq!(w, ImageWindow::new(40, 40, 20, 20));
    }

    #[test]
    fn test_spot_window_side_rounds_up() {
        let img = grid(1.0);
        // 10.0 * 1.0 * 0.31 = 3.1, so 4 px.
        let w = spot_window(&img, (50.0, 50.0), 10.0, 0.31).unwrap();
        assert_eq!(w.width, 4);
        assert_eq!(w.height, 4);
    }

    #[test]
    fn test_spot_window_clips_at_the_top_left_corner() {
        let img = grid(1.0);
        let w = spot_window(&img, (1.0, 1.0), 10.0, 1.0).unwrap();
        // Unclipped window would be (-4, -4, 10, 10).
        assert_eq!(w, ImageWindow::new(0, 0, 6, 6));
    }

    #[test]
    fn test_spot_window_clips_at_the_bottom_right_corner() {
        let img = grid(1.0);
        let w = spot_window(&img, (99.0, 99.0), 10.0, 1.0).unwrap();
        assert_eq!(w, ImageWindow::new(94, 94, 6, 6));
    }

    #[test]
    fn test_spot_window_off_image_errors() {
        let img = grid(1.0);
        assert!(matches!(
            spot_window(&img, (500.0, 500.0), 10.0, 1.0),
            Err(BixverseErrors::ImageWindowOutOfBounds { .. })
        ));
        assert!(matches!(
            spot_window(&img, (-50.0, 50.0), 10.0, 1.0),
            Err(BixverseErrors::ImageWindowOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_spot_window_never_degenerates_to_zero_pixels() {
        let img = grid(1.0);
        let w = spot_window(&img, (50.0, 50.0), 0.0, 1.0).unwrap();
        assert_eq!((w.width, w.height), (1, 1));
    }

    #[test]
    fn test_extract_spot_tiles_content_is_correct() {
        let img = grid(1.0);
        let tiles = extract_spot_tiles(&img, &[(10.0, 20.0), (50.0, 60.0)], 4.0, 1.0).unwrap();
        assert_eq!(tiles.len(), 2);
        assert_eq!((tiles[0].width, tiles[0].height), (4, 4));
        // Window starts at (8, 18), so the first pixel is [8, 18, 0].
        assert_eq!(&tiles[0].data[0..3], &[8, 18, 0]);
        assert_eq!(&tiles[1].data[0..3], &[48, 58, 0]);
    }

    #[test]
    fn test_extract_spot_tiles_preserves_input_order() {
        let img = grid(1.0);
        let coords: Vec<(f32, f32)> = (10..90).map(|i| (i as f32, i as f32)).collect();
        let tiles = extract_spot_tiles(&img, &coords, 4.0, 1.0).unwrap();
        assert_eq!(tiles.len(), coords.len());
        for (t, c) in tiles.iter().zip(coords.iter()) {
            assert_eq!(t.data[0], c.0 as u8 - 2);
        }
    }
}
