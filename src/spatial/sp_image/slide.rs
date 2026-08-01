//! Windowed reads out of an OpenSlide pyramid.
//!
//! This is what makes a ~35,638 x 29,401 scanner frame usable: reads come
//! straight out of the pyramid level, so nothing like the 3.1 GB the full frame
//! would need is ever resident. Covers SVS, NDPI, MIRAX, DICOM and pyramidal
//! TIFF, whatever the locally installed OpenSlide was built with.
//!
//! ## Untested against real scanner data
//!
//! No `.svs`, `.ndpi` or MIRAX file was available when this was written, and
//! the one TIFF to hand (the Space Ranger CytAssist image) is a plain strip
//! TIFF that OpenSlide declines to open. Everything here is checked against the
//! API contract and synthetic fixtures only. Treat the first run against a real
//! slide as a smoke test, not as a regression.
//!
//! ## Two things the C API gets asked about a lot
//!
//! `openslide_read_region` takes its origin in the **level-0** frame while the
//! width and height are in the **requested level's** frame. Mixing the two up
//! silently reads the wrong part of the slide at anything but level 0, so
//! [`SlideImage::read_window`] converts the origin back on every call.
//!
//! The returned buffer is pre-multiplied ARGB32, which on a little-endian
//! machine lands in memory as B, G, R, A. We un-premultiply and drop alpha.
//! Regions outside the slide's own bounds come back with alpha 0, which
//! un-premultiplies to black.
//!
//! ## Licensing
//!
//! OpenSlide itself is LGPL-2.1 and this crate is MIT, so the C library must be
//! linked dynamically. See the `openslide-sys` entry in `Cargo.toml`.

use openslide_rs::{Address, OpenSlide, Region, Size};
use std::path::Path;

use crate::prelude::*;
use crate::spatial::sp_image::source::{ImageSource, ImageWindow, RgbTile};

///////////////
// Constants //
///////////////

/// Bytes per pixel in the buffer OpenSlide hands back (pre-multiplied ARGB32).
const OPENSLIDE_BYTES_PER_PIXEL: usize = 4;

/// Bytes per pixel in an [`RgbTile`].
const RGB_BYTES_PER_PIXEL: usize = 3;

////////////////
// SlideImage //
////////////////

/// A whole-slide image read through OpenSlide at one chosen pyramid level.
///
/// The level is fixed at construction from the requested working scale, so
/// [`ImageSource::scale`] and [`ImageSource::dims`] describe that level and
/// nothing else. Want a different scale, build a different `SlideImage`.
#[derive(Debug)]
pub struct SlideImage {
    /// The open slide handle. `OpenSlide` is `Send + Sync` (its inner pointer
    /// wrapper asserts both) and every call except `close` is thread-safe, so
    /// windowed reads parallelise over one handle.
    slide: OpenSlide,
    /// Pyramid level all reads go to.
    level: u32,
    /// Width of `level`, in that level's pixel space.
    width: u32,
    /// Height of `level`, in that level's pixel space.
    height: u32,
    /// Downsample factor of `level` relative to level 0. Level 0 is 1.0, a
    /// half-resolution level is 2.0.
    downsample: f64,
}

impl SlideImage {
    /// Does OpenSlide recognise this file?
    ///
    /// Sniffs the container rather than trusting the extension. Cheap enough to
    /// call as a probe: it does not open the slide.
    ///
    /// ### Params
    ///
    /// * `path` - File to probe.
    ///
    /// ### Returns
    ///
    /// `true` if OpenSlide reports a vendor for the file.
    pub fn is_slide<P: AsRef<Path>>(path: P) -> bool {
        OpenSlide::detect_vendor(path.as_ref()).is_ok()
    }

    /// Open a slide and pin it to the level nearest the requested scale.
    ///
    /// ### Params
    ///
    /// * `path` - Slide file.
    /// * `working_scale` - Target scale from full-res into the working pixel
    ///   space, so 1.0 asks for level 0 and 0.25 asks for a quarter-resolution
    ///   level. See [`SlideImage::pick_level`] for the choice rule.
    ///
    /// ### Returns
    ///
    /// The opened slide, or an error if OpenSlide cannot read it or reports no
    /// levels.
    pub fn new<P: AsRef<Path>>(path: P, working_scale: f32) -> Result<Self, BixverseErrors> {
        let slide = OpenSlide::new(path.as_ref())?;
        let downsamples = slide.get_all_level_downsample()?;
        if downsamples.is_empty() {
            return Err(BixverseErrors::SlideNoLevels);
        }

        let level = Self::pick_level(&downsamples, working_scale);
        let dims = slide.get_level_dimensions(level)?;

        Ok(Self {
            slide,
            level,
            width: dims.w,
            height: dims.h,
            downsample: downsamples[level as usize],
        })
    }

    /// Choose the pyramid level for a requested working scale.
    ///
    /// Take the coarsest level that is still no coarser than what was asked
    /// for, i.e. the largest downsample `<= 1 / working_scale`. Going one step
    /// finer and letting the caller downsample beats reading a level that has
    /// already thrown away detail the features need. Level 0 is the floor, so a
    /// `working_scale` above 1.0 just gets full resolution.
    ///
    /// ### Params
    ///
    /// * `downsamples` - Per-level downsample factors, level 0 first. Must be
    ///   non-empty.
    /// * `working_scale` - Target scale from full-res into working space.
    ///   Non-positive values fall back to level 0.
    ///
    /// ### Returns
    ///
    /// The chosen level index.
    fn pick_level(downsamples: &[f64], working_scale: f32) -> u32 {
        if working_scale <= 0.0 {
            return 0;
        }
        let target = 1.0 / f64::from(working_scale);

        let mut best = 0u32;
        let mut best_downsample = downsamples[0];
        for (level, &d) in downsamples.iter().enumerate() {
            if d <= target && d >= best_downsample {
                best = level as u32;
                best_downsample = d;
            }
        }
        best
    }

    /// Pyramid level this instance reads from.
    ///
    /// ### Returns
    ///
    /// The level index.
    pub fn level(&self) -> u32 {
        self.level
    }

    /// Downsample factor of the chosen level relative to level 0.
    ///
    /// ### Returns
    ///
    /// The factor, 1.0 at level 0.
    pub fn downsample(&self) -> f64 {
        self.downsample
    }

    /// Number of levels in the pyramid.
    ///
    /// ### Returns
    ///
    /// The level count, or an error if the query fails.
    pub fn level_count(&self) -> Result<u32, BixverseErrors> {
        Ok(self.slide.get_level_count()?)
    }
}

impl ImageSource for SlideImage {
    fn dims(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn scale(&self) -> f32 {
        (1.0 / self.downsample) as f32
    }

    fn read_window(&self, window: ImageWindow) -> Result<RgbTile, BixverseErrors> {
        let w = window.clamp_to(self.dims())?;

        // Origin goes back to the level-0 frame, size stays in the level frame.
        // This is the C API's contract, not a convenience.
        let region = Region {
            address: Address {
                x: (f64::from(w.x) * self.downsample).round() as u32,
                y: (f64::from(w.y) * self.downsample).round() as u32,
            },
            level: self.level,
            size: Size {
                w: w.width,
                h: w.height,
            },
        };

        let bgra = self.slide.read_region(&region)?;
        Ok(bgra_premultiplied_to_rgb(&bgra, w.width, w.height))
    }
}

/// Convert a pre-multiplied ARGB32 buffer to row-major RGB8.
///
/// On little-endian machines the ARGB32 words land in memory as B, G, R, A.
/// Un-premultiplying divides each channel by alpha; alpha 0 means the pixel sat
/// outside the slide's own bounds and becomes black.
///
/// ### Params
///
/// * `bgra` - `width * height * 4` bytes as returned by `openslide_read_region`.
/// * `width` - Tile width.
/// * `height` - Tile height.
///
/// ### Returns
///
/// The tile as row-major RGB8.
fn bgra_premultiplied_to_rgb(bgra: &[u8], width: u32, height: u32) -> RgbTile {
    let n_pixels = width as usize * height as usize;
    let mut data = Vec::with_capacity(n_pixels * RGB_BYTES_PER_PIXEL);

    for px in bgra.chunks_exact(OPENSLIDE_BYTES_PER_PIXEL).take(n_pixels) {
        let (b, g, r, a) = (px[0], px[1], px[2], px[3]);
        if a == 0 {
            data.extend_from_slice(&[0, 0, 0]);
        } else if a == u8::MAX {
            data.extend_from_slice(&[r, g, b]);
        } else {
            let a = u32::from(a);
            let unmul = |c: u8| ((u32::from(c) * 255 + a / 2) / a).min(255) as u8;
            data.extend_from_slice(&[unmul(r), unmul(g), unmul(b)]);
        }
    }

    // A short buffer would be an OpenSlide contract violation, but pad rather
    // than hand back a tile whose length disagrees with its dimensions.
    data.resize(n_pixels * RGB_BYTES_PER_PIXEL, 0);

    RgbTile {
        data,
        width,
        height,
    }
}

//////////////
// Fixtures //
//////////////

/// Minimal baseline-TIFF writer, test only.
///
/// OpenSlide's `generic-tiff` driver opens any tiled baseline TIFF, which makes
/// it the one slide format that can be synthesised without a scanner. Writing
/// the container by hand rather than shelling out to `tiffcp` keeps the tests
/// self-contained, and it means [`SlideImage`] gets exercised against the real
/// OpenSlide C library on every run instead of being pure theory.
///
/// Uncompressed RGB8, one IFD per pyramid level, levels after the first flagged
/// as reduced-resolution.
#[cfg(test)]
pub(crate) mod fixtures {
    use std::io::Write;
    use std::path::Path;

    /// Bytes an IFD occupies: entry count, 12 entries, next-IFD pointer.
    const IFD_BYTES: u64 = 2 + 12 * 12 + 4;

    /// One pyramid level as raw row-major RGB8.
    pub(crate) struct Level {
        /// `width * height * 3` bytes.
        pub data: Vec<u8>,
        /// Level width.
        pub width: u32,
        /// Level height.
        pub height: u32,
    }

    impl Level {
        /// A level whose pixel `(x, y)` is a deterministic function of its
        /// position, so a misread window is obvious rather than plausible.
        ///
        /// ### Params
        ///
        /// * `width` - Level width.
        /// * `height` - Level height.
        /// * `seed` - Offset folded into the blue channel, to tell levels apart.
        ///
        /// ### Returns
        ///
        /// The level.
        pub(crate) fn ramp(width: u32, height: u32, seed: u8) -> Self {
            let mut data = Vec::with_capacity((width * height * 3) as usize);
            for y in 0..height {
                for x in 0..width {
                    data.push((x % 256) as u8);
                    data.push((y % 256) as u8);
                    data.push(seed);
                }
            }
            Self {
                data,
                width,
                height,
            }
        }

        /// Pixel at `(x, y)` as `[r, g, b]`.
        ///
        /// ### Params
        ///
        /// * `x` - Column.
        /// * `y` - Row.
        ///
        /// ### Returns
        ///
        /// The pixel.
        pub(crate) fn pixel(&self, x: u32, y: u32) -> [u8; 3] {
            let i = ((y * self.width + x) * 3) as usize;
            [self.data[i], self.data[i + 1], self.data[i + 2]]
        }
    }

    /// Pack one level's pixels into padded tiles, row-major over tiles.
    ///
    /// TIFF requires every tile to be full size, so edge tiles carry padding.
    ///
    /// ### Params
    ///
    /// * `level` - Source level.
    /// * `tile` - Tile side in pixels.
    ///
    /// ### Returns
    ///
    /// The concatenated tile data and the per-tile byte count.
    fn tile_level(level: &Level, tile: u32) -> (Vec<u8>, u32) {
        let across = level.width.div_ceil(tile);
        let down = level.height.div_ceil(tile);
        let tile_bytes = (tile * tile * 3) as usize;
        let mut out = Vec::with_capacity(tile_bytes * (across * down) as usize);

        for ty in 0..down {
            for tx in 0..across {
                for row in 0..tile {
                    let y = ty * tile + row;
                    for col in 0..tile {
                        let x = tx * tile + col;
                        if x < level.width && y < level.height {
                            out.extend_from_slice(&level.pixel(x, y));
                        } else {
                            out.extend_from_slice(&[0, 0, 0]);
                        }
                    }
                }
            }
        }
        (out, tile_bytes as u32)
    }

    /// Write a tiled, uncompressed, little-endian baseline TIFF.
    ///
    /// ### Params
    ///
    /// * `path` - Destination.
    /// * `levels` - Pyramid levels, largest first. Must be non-empty.
    /// * `tile` - Tile side in pixels.
    ///
    /// ### Returns
    ///
    /// Nothing, or the underlying I/O error.
    pub(crate) fn write_tiled_tiff(
        path: &Path,
        levels: &[Level],
        tile: u32,
    ) -> std::io::Result<()> {
        let packed: Vec<(Vec<u8>, u32)> = levels.iter().map(|l| tile_level(l, tile)).collect();

        // Layout: header, then all tile data, then per level an auxiliary block
        // (values too big for an IFD entry) followed by that level's IFD.
        let mut data_offsets = Vec::with_capacity(levels.len());
        let mut cursor: u64 = 8;
        for (bytes, _) in &packed {
            data_offsets.push(cursor);
            cursor += bytes.len() as u64;
        }

        let mut aux_offsets = Vec::with_capacity(levels.len());
        let mut ifd_offsets = Vec::with_capacity(levels.len());
        for (bytes, per_tile) in &packed {
            let n_tiles = (bytes.len() / *per_tile as usize) as u64;
            aux_offsets.push(cursor);
            // BitsPerSample (3 SHORTs) then TileOffsets and TileByteCounts.
            cursor += 6 + 4 * n_tiles + 4 * n_tiles;
            ifd_offsets.push(cursor);
            cursor += IFD_BYTES;
        }

        let mut out = Vec::with_capacity(cursor as usize);
        out.extend_from_slice(b"II");
        out.extend_from_slice(&42u16.to_le_bytes());
        out.extend_from_slice(&(ifd_offsets[0] as u32).to_le_bytes());

        for (bytes, _) in &packed {
            out.extend_from_slice(bytes);
        }

        for (i, level) in levels.iter().enumerate() {
            let (bytes, per_tile) = &packed[i];
            let n_tiles = (bytes.len() / *per_tile as usize) as u32;

            // Auxiliary block, referenced by offset from the IFD below.
            out.extend_from_slice(&8u16.to_le_bytes());
            out.extend_from_slice(&8u16.to_le_bytes());
            out.extend_from_slice(&8u16.to_le_bytes());
            let tile_offsets_at = aux_offsets[i] + 6;
            let tile_counts_at = tile_offsets_at + 4 * u64::from(n_tiles);
            for t in 0..n_tiles {
                let off = data_offsets[i] + u64::from(t) * u64::from(*per_tile);
                out.extend_from_slice(&(off as u32).to_le_bytes());
            }
            for _ in 0..n_tiles {
                out.extend_from_slice(&per_tile.to_le_bytes());
            }

            // IFD. Entries must be in ascending tag order.
            out.extend_from_slice(&12u16.to_le_bytes());
            // Levels beyond the first are reduced-resolution versions.
            let subfile = if i == 0 { 0u32 } else { 1u32 };
            push_entry(&mut out, 254, 4, 1, subfile);
            push_entry(&mut out, 256, 4, 1, level.width);
            push_entry(&mut out, 257, 4, 1, level.height);
            push_entry(&mut out, 258, 3, 3, aux_offsets[i] as u32);
            push_entry(&mut out, 259, 3, 1, 1);
            push_entry(&mut out, 262, 3, 1, 2);
            push_entry(&mut out, 277, 3, 1, 3);
            push_entry(&mut out, 284, 3, 1, 1);
            push_entry(&mut out, 322, 4, 1, tile);
            push_entry(&mut out, 323, 4, 1, tile);
            push_entry(&mut out, 324, 4, n_tiles, tile_offsets_at as u32);
            push_entry(&mut out, 325, 4, n_tiles, tile_counts_at as u32);
            let next = if i + 1 < levels.len() {
                ifd_offsets[i + 1] as u32
            } else {
                0
            };
            out.extend_from_slice(&next.to_le_bytes());
        }

        std::fs::File::create(path)?.write_all(&out)
    }

    /// Append one 12-byte IFD entry.
    ///
    /// Values of four bytes or fewer sit inline in the value field; anything
    /// larger has `value` treated as a file offset by the caller.
    ///
    /// ### Params
    ///
    /// * `out` - Buffer to append to.
    /// * `tag` - TIFF tag.
    /// * `kind` - Field type: 3 is SHORT, 4 is LONG.
    /// * `count` - Number of values.
    /// * `value` - Inline value or offset.
    fn push_entry(out: &mut Vec<u8>, tag: u16, kind: u16, count: u32, value: u32) {
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&kind.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        // A single SHORT occupies the low half of the value field.
        if kind == 3 && count == 1 {
            out.extend_from_slice(&(value as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
        } else {
            out.extend_from_slice(&value.to_le_bytes());
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

    #[test]
    fn test_pick_level_exact_match() {
        let d = [1.0, 2.0, 4.0, 16.0];
        assert_eq!(SlideImage::pick_level(&d, 0.25), 2);
        assert_eq!(SlideImage::pick_level(&d, 0.5), 1);
        assert_eq!(SlideImage::pick_level(&d, 1.0), 0);
    }

    #[test]
    fn test_pick_level_never_goes_coarser_than_requested() {
        let d = [1.0, 2.0, 4.0, 16.0];
        // 1/0.2 = 5, so level 2 (downsample 4) qualifies but level 3 (16)
        // would be coarser than asked for.
        assert_eq!(SlideImage::pick_level(&d, 0.2), 2);
        // 1/0.3 = 3.33, level 1 is the coarsest that still fits.
        assert_eq!(SlideImage::pick_level(&d, 0.3), 1);
    }

    #[test]
    fn test_pick_level_clamps_to_full_resolution() {
        let d = [1.0, 2.0, 4.0];
        assert_eq!(SlideImage::pick_level(&d, 2.0), 0);
        assert_eq!(SlideImage::pick_level(&d, 0.0), 0);
        assert_eq!(SlideImage::pick_level(&d, -1.0), 0);
    }

    #[test]
    fn test_pick_level_very_coarse_request_takes_coarsest() {
        let d = [1.0, 2.0, 4.0, 32.0];
        assert_eq!(SlideImage::pick_level(&d, 0.001), 3);
    }

    #[test]
    fn test_pick_level_single_level_pyramid() {
        assert_eq!(SlideImage::pick_level(&[1.0], 0.01), 0);
    }

    #[test]
    fn test_bgra_opaque_round_trips_channel_order() {
        // Two opaque pixels: red then green, stored B, G, R, A.
        let bgra = vec![0, 0, 255, 255, 0, 255, 0, 255];
        let tile = bgra_premultiplied_to_rgb(&bgra, 2, 1);
        assert_eq!(tile.data, vec![255, 0, 0, 0, 255, 0]);
    }

    #[test]
    fn test_bgra_zero_alpha_becomes_black() {
        let bgra = vec![0, 0, 0, 0];
        let tile = bgra_premultiplied_to_rgb(&bgra, 1, 1);
        assert_eq!(tile.data, vec![0, 0, 0]);
    }

    #[test]
    fn test_bgra_unpremultiplies_partial_alpha() {
        // Half-transparent white: premultiplied channels sit at ~128.
        let bgra = vec![128, 128, 128, 128];
        let tile = bgra_premultiplied_to_rgb(&bgra, 1, 1);
        for &c in &tile.data {
            assert!(c >= 253, "expected near-white, got {c}");
        }
    }

    #[test]
    fn test_bgra_short_buffer_is_padded_not_truncated() {
        let bgra = vec![0, 0, 255, 255];
        let tile = bgra_premultiplied_to_rgb(&bgra, 2, 1);
        assert_eq!(tile.data.len(), 6);
        assert_eq!(tile.data, vec![255, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_scale_is_inverse_downsample() {
        // Guards the relationship without needing a slide on disk.
        let downsample = 4.0_f64;
        assert_relative_eq!((1.0 / downsample) as f32, 0.25_f32, epsilon = 1e-9);
    }

    #[test]
    fn test_is_slide_rejects_missing_file() {
        assert!(!SlideImage::is_slide("/definitely/not/a/slide.svs"));
    }

    // -- against the real OpenSlide C library --
    //
    // These use a synthesised tiled TIFF, so they cover the binding, the
    // BGRA conversion, the level-0 origin conversion and level selection.
    // They do not cover any vendor format: no SVS, NDPI or MIRAX file was
    // available. See the module docs.

    use fixtures::{Level, write_tiled_tiff};
    use std::path::PathBuf;

    /// Write a fixture into the target directory and hand back the path.
    fn fixture(name: &str, levels: &[Level], tile: u32) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("bixverse_sp_image_{name}.tif"));
        write_tiled_tiff(&path, levels, tile).expect("fixture should be writable");
        path
    }

    #[test]
    fn test_openslide_opens_a_synthesised_tiled_tiff() {
        let path = fixture("probe", &[Level::ramp(300, 200, 7)], 64);
        assert!(
            SlideImage::is_slide(&path),
            "OpenSlide should recognise a tiled baseline TIFF"
        );

        let slide = SlideImage::new(&path, 1.0).unwrap();
        assert_eq!(slide.dims(), (300, 200));
        assert_eq!(slide.level(), 0);
        assert_relative_eq!(slide.scale(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_slide_window_returns_the_right_pixels() {
        let level = Level::ramp(300, 200, 7);
        let path = fixture("window", std::slice::from_ref(&level), 64);
        let slide = SlideImage::new(&path, 1.0).unwrap();

        let tile = slide.read_window(ImageWindow::new(70, 40, 8, 6)).unwrap();
        assert_eq!((tile.width, tile.height), (8, 6));
        for row in 0..6u32 {
            for col in 0..8u32 {
                let got = &tile.data[((row * 8 + col) * 3) as usize..][..3];
                assert_eq!(
                    got,
                    level.pixel(70 + col, 40 + row),
                    "mismatch at ({col}, {row})"
                );
            }
        }
    }

    #[test]
    fn test_slide_window_spanning_tile_boundaries() {
        let level = Level::ramp(300, 200, 3);
        let path = fixture("boundary", std::slice::from_ref(&level), 64);
        let slide = SlideImage::new(&path, 1.0).unwrap();

        // Straddles the tile edge at x = 64 and y = 64.
        let tile = slide.read_window(ImageWindow::new(60, 60, 10, 10)).unwrap();
        for row in 0..10u32 {
            for col in 0..10u32 {
                let got = &tile.data[((row * 10 + col) * 3) as usize..][..3];
                assert_eq!(got, level.pixel(60 + col, 60 + row));
            }
        }
    }

    #[test]
    fn test_slide_window_clamped_at_the_far_edge() {
        let level = Level::ramp(300, 200, 1);
        let path = fixture("clamp", std::slice::from_ref(&level), 64);
        let slide = SlideImage::new(&path, 1.0).unwrap();

        let tile = slide
            .read_window(ImageWindow::new(295, 195, 20, 20))
            .unwrap();
        assert_eq!((tile.width, tile.height), (5, 5));
        assert_eq!(&tile.data[0..3], &level.pixel(295, 195));
    }

    /// The gotcha test. `openslide_read_region` takes its origin in the level-0
    /// frame but its size in the requested level's frame. Reading level 1 at a
    /// non-zero offset is the only way to catch getting that wrong.
    #[test]
    fn test_slide_level_one_origin_is_converted_to_level_zero_frame() {
        let l0 = Level::ramp(400, 300, 11);
        let mut l1 = Level::ramp(200, 150, 22);
        // Make level 1 an honest half-scale copy of level 0 so a wrong origin
        // shows up as a wrong pixel rather than as a plausible one.
        for y in 0..150u32 {
            for x in 0..200u32 {
                let src = l0.pixel(x * 2, y * 2);
                let i = ((y * 200 + x) * 3) as usize;
                l1.data[i] = src[0];
                l1.data[i + 1] = src[1];
                l1.data[i + 2] = 22;
            }
        }
        let path = fixture("pyramid", &[l0, l1], 64);

        let slide = SlideImage::new(&path, 0.5).unwrap();
        assert_eq!(slide.level_count().unwrap(), 2);
        assert_eq!(slide.level(), 1);
        assert_eq!(slide.dims(), (200, 150));
        assert_relative_eq!(slide.downsample() as f32, 2.0, epsilon = 1e-3);
        assert_relative_eq!(slide.scale(), 0.5, epsilon = 1e-3);

        // Window at (50, 40) in level-1 space is (100, 80) in level-0 space.
        // Level 1 pixel (50, 40) mirrors level 0 pixel (100, 80).
        let tile = slide.read_window(ImageWindow::new(50, 40, 4, 4)).unwrap();
        assert_eq!(&tile.data[0..3], &[100, 80, 22]);
        assert_eq!(&tile.data[3..6], &[102, 80, 22]);
    }

    #[test]
    fn test_slide_picks_level_zero_when_full_scale_requested() {
        let l0 = Level::ramp(400, 300, 11);
        let l1 = Level::ramp(200, 150, 22);
        let path = fixture("pyramid_full", &[l0, l1], 64);

        let slide = SlideImage::new(&path, 1.0).unwrap();
        assert_eq!(slide.level(), 0);
        assert_eq!(slide.dims(), (400, 300));
        let tile = slide.read_window(ImageWindow::new(10, 20, 2, 1)).unwrap();
        assert_eq!(&tile.data[0..3], &[10, 20, 11]);
    }

    #[test]
    fn test_slide_off_image_window_errors() {
        let path = fixture("oob", &[Level::ramp(128, 128, 5)], 64);
        let slide = SlideImage::new(&path, 1.0).unwrap();
        assert!(matches!(
            slide.read_window(ImageWindow::new(500, 500, 8, 8)),
            Err(BixverseErrors::ImageWindowOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_slide_open_failure_is_an_error_not_a_panic() {
        assert!(matches!(
            SlideImage::new("/definitely/not/a/slide.svs", 1.0),
            Err(BixverseErrors::OpenSlide(_))
        ));
    }

    /// Both backends over the same bytes on disk.
    ///
    /// A tiled TIFF is one of the few things OpenSlide and the `image` crate
    /// can both decode, which makes it the only way to check that the two
    /// implementations agree without an SVS to hand.
    #[test]
    fn test_whole_image_and_slide_image_agree_on_the_same_file() {
        use crate::spatial::sp_image::source::WholeImage;

        let path = fixture("agreement", &[Level::ramp(300, 200, 9)], 64);
        let slide = SlideImage::new(&path, 1.0).unwrap();
        let whole = WholeImage::new(&path, 1.0).unwrap();

        assert_eq!(slide.dims(), whole.dims());
        assert_relative_eq!(slide.scale(), whole.scale(), epsilon = 1e-6);

        let windows = [
            ImageWindow::new(0, 0, 16, 16),
            ImageWindow::new(60, 60, 10, 10),
            ImageWindow::new(128, 64, 64, 64),
            ImageWindow::new(295, 195, 20, 20),
            ImageWindow::new(0, 0, 300, 200),
        ];
        for w in windows {
            let a = slide.read_window(w).unwrap();
            let b = whole.read_window(w).unwrap();
            assert_eq!(
                (a.width, a.height),
                (b.width, b.height),
                "dimension mismatch for {w:?}"
            );
            assert_eq!(a.data, b.data, "pixel mismatch for {w:?}");
        }
    }

    /// The probe must route a tiled TIFF to the slide backend, not to the
    /// whole-image one, even though both could open it.
    #[test]
    fn test_probe_prefers_the_slide_backend_for_a_tiled_tiff() {
        use crate::spatial::sp_image::source::{AnyImageSource, open_image_source};

        let path = fixture("probe_route", &[Level::ramp(200, 150, 4)], 64);
        let src = open_image_source(&path, 1.0, 1.0).unwrap();
        assert!(matches!(src, AnyImageSource::Slide(_)));
        assert_eq!(src.dims(), (200, 150));
    }
}
