//! The windowed-read boundary.
//!
//! Everything above this file is resolution- and format-agnostic. A hires PNG,
//! a CytAssist TIFF and a scanner SVS differ only in how a window is fetched,
//! so that is the only thing [`ImageSource`] exposes.
//!
//! Two implementations back the trait. [`WholeImage`] decodes a file once and
//! crops out of the resident buffer, which is right for the Space Ranger
//! outputs (the 2000x1650 hires PNG is about 10 MB decoded). [`SlideImage`] in
//! `slide.rs` reads windows straight out of a pyramid and never holds the whole
//! frame.
//!
//! [`SlideImage`]: crate::spatial::sp_image::slide::SlideImage

use image::GenericImageView;
use std::path::Path;

use crate::prelude::*;
use crate::spatial::sp_image::slide::SlideImage;

/////////////
// Windows //
/////////////

/// A window in the backing image's own pixel space.
///
/// Not full-res pixels. The caller converts, or lets [`ImageSource::scale`] do
/// it. See the module docs of `sp_image` for the index-space contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageWindow {
    /// Left edge, in the backing image's pixel space.
    pub x: u32,
    /// Top edge, in the backing image's pixel space.
    pub y: u32,
    /// Window width in pixels.
    pub width: u32,
    /// Window height in pixels.
    pub height: u32,
}

impl ImageWindow {
    /// Build a window.
    ///
    /// ### Params
    ///
    /// * `x` - Left edge in the backing image's pixel space.
    /// * `y` - Top edge in the backing image's pixel space.
    /// * `width` - Width in pixels.
    /// * `height` - Height in pixels.
    ///
    /// ### Returns
    ///
    /// The window.
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Intersect the window with a `dims`-sized image.
    ///
    /// Partly overlapping windows come back smaller. A window with no overlap
    /// at all is an error rather than a zero-sized tile, because every
    /// downstream feature is undefined on zero pixels.
    ///
    /// ### Params
    ///
    /// * `dims` - `(width, height)` of the backing image.
    ///
    /// ### Returns
    ///
    /// The clamped window, or [`BixverseErrors::ImageWindowOutOfBounds`] if the
    /// intersection is empty.
    pub fn clamp_to(&self, dims: (u32, u32)) -> Result<Self, BixverseErrors> {
        let (img_w, img_h) = dims;
        let x1 = self.x.saturating_add(self.width).min(img_w);
        let y1 = self.y.saturating_add(self.height).min(img_h);
        let x0 = self.x.min(x1);
        let y0 = self.y.min(y1);

        if x1 <= x0 || y1 <= y0 {
            return Err(BixverseErrors::ImageWindowOutOfBounds {
                x: self.x,
                y: self.y,
                width: self.width,
                height: self.height,
                img_width: img_w,
                img_height: img_h,
            });
        }

        Ok(Self {
            x: x0,
            y: y0,
            width: x1 - x0,
            height: y1 - y0,
        })
    }
}

/// Row-major RGB8. Three bytes per pixel, no padding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbTile {
    /// `width * height * 3` bytes, row-major, R then G then B.
    pub data: Vec<u8>,
    /// Tile width in pixels.
    pub width: u32,
    /// Tile height in pixels.
    pub height: u32,
}

impl RgbTile {
    /// Number of pixels in the tile.
    ///
    /// ### Returns
    ///
    /// `width * height`.
    pub fn n_pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Iterate over pixels as `[r, g, b]`.
    ///
    /// ### Returns
    ///
    /// An iterator over three-byte slices in row-major order.
    pub fn pixels(&self) -> impl Iterator<Item = &[u8]> {
        self.data.chunks_exact(3)
    }
}

///////////
// Trait //
///////////

/// A source that can hand back an RGB window of a histology image.
///
/// The whole point of the abstraction: a 2000-pixel PNG and a 35,638-pixel
/// scanner frame differ only in how the window is fetched, so tiling, colour
/// deconvolution and texture never learn which one they are looking at.
pub trait ImageSource: Send + Sync {
    /// Dimensions of the backing image, `(width, height)`, in its own pixel
    /// space.
    fn dims(&self) -> (u32, u32);

    /// Scale from full-res coordinates into this image's pixel space.
    ///
    /// A full-res coordinate `x` lands at `x * scale()`. So the hires Space
    /// Ranger PNG returns its `tissue_hires_scalef` (0.0561 for the bundled
    /// example) and a level-0 slide read returns 1.0.
    fn scale(&self) -> f32;

    /// Read a window and return it as row-major RGB8.
    ///
    /// The window is in this image's pixel space and gets clamped to the image
    /// bounds, so the returned tile can be smaller than asked for.
    ///
    /// ### Params
    ///
    /// * `window` - The region to fetch, in this image's pixel space.
    ///
    /// ### Returns
    ///
    /// The tile, or an error if the window misses the image entirely or the
    /// backing reader fails.
    fn read_window(&self, window: ImageWindow) -> Result<RgbTile, BixverseErrors>;
}

////////////////
// WholeImage //
////////////////

/// A fully decoded image held in memory, cropped on demand.
///
/// Right for anything that fits comfortably in RAM: the Space Ranger hires and
/// lowres PNGs, the aligned JPEGs. Decoding happens once in [`WholeImage::new`]
/// and every `read_window` after that is a memcpy out of the resident buffer.
///
/// Only PNG and JPEG are compiled in, because that is what the `image`
/// dependency enables. Anything else needs the slide path.
#[derive(Debug, Clone)]
pub struct WholeImage {
    /// Decoded RGB8 buffer, `width * height * 3` bytes, row-major.
    data: Vec<u8>,
    /// Image width in pixels.
    width: u32,
    /// Image height in pixels.
    height: u32,
    /// Scale from full-res coordinates into this image's pixel space.
    scale: f32,
}

impl WholeImage {
    /// Decode an image file in full.
    ///
    /// The scale factor cannot be recovered from the file: a PNG does not know
    /// it is a 5.6% downsample of something bigger. Read it from
    /// `scalefactors_json.json` and pass it in.
    ///
    /// ### Params
    ///
    /// * `path` - Path to a PNG or JPEG.
    /// * `scale` - Scale from full-res coordinates into this image's pixel
    ///   space, e.g. `tissue_hires_scalef`. Use 1.0 if the file already is
    ///   full-res.
    ///
    /// An RGBA file gets composited onto white rather than having its alpha
    /// dropped. `into_rgb8` discards the channel, which turns a transparent
    /// region black, and black is the maximum of the optical density
    /// transform. An RGBA PNG or TIFF out of QuPath or ImageJ is the usual
    /// source of one.
    ///
    /// ### Returns
    ///
    /// The decoded image, or [`BixverseErrors::ImageDecode`] if the file cannot
    /// be read.
    pub fn new<P: AsRef<Path>>(path: P, scale: f32) -> Result<Self, BixverseErrors> {
        let decoded = image::open(path.as_ref())?;
        let (width, height) = decoded.dimensions();
        let data = if decoded.color().has_alpha() {
            rgba_onto_white(&decoded.into_rgba8())
        } else {
            decoded.into_rgb8().into_raw()
        };
        Ok(Self {
            data,
            width,
            height,
            scale,
        })
    }

    /// Build directly from a decoded RGB8 buffer.
    ///
    /// Used by the tests to avoid touching the filesystem. Panics in debug if
    /// the buffer length disagrees with the dimensions.
    ///
    /// ### Params
    ///
    /// * `data` - `width * height * 3` bytes, row-major RGB8.
    /// * `width` - Image width.
    /// * `height` - Image height.
    /// * `scale` - Scale from full-res coordinates into this image's space.
    ///
    /// ### Returns
    ///
    /// The image.
    pub fn from_rgb8(data: Vec<u8>, width: u32, height: u32, scale: f32) -> Self {
        debug_assert_eq!(data.len(), width as usize * height as usize * 3);
        Self {
            data,
            width,
            height,
            scale,
        }
    }
}

impl ImageSource for WholeImage {
    fn dims(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn scale(&self) -> f32 {
        self.scale
    }

    fn read_window(&self, window: ImageWindow) -> Result<RgbTile, BixverseErrors> {
        let w = window.clamp_to(self.dims())?;
        let row_stride = self.width as usize * 3;
        let tile_stride = w.width as usize * 3;
        let mut data = Vec::with_capacity(tile_stride * w.height as usize);

        for row in w.y..(w.y + w.height) {
            let start = row as usize * row_stride + w.x as usize * 3;
            data.extend_from_slice(&self.data[start..start + tile_stride]);
        }

        Ok(RgbTile {
            data,
            width: w.width,
            height: w.height,
        })
    }
}

/// Flatten an RGBA8 buffer onto a white background.
///
/// Straight `src-over` on non-premultiplied channels:
/// `out = (c a + 255 (255 - a) + 127) / 255`. White because brightfield
/// background is white, and because black sits at the ceiling of the optical
/// density transform.
///
/// ### Params
///
/// * `rgba` - Decoded RGBA8 image.
///
/// ### Returns
///
/// Row-major RGB8, three bytes per pixel.
fn rgba_onto_white(rgba: &image::RgbaImage) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgba.len() / 4 * 3);
    for px in rgba.chunks_exact(4) {
        let a = u32::from(px[3]);
        let bg = 255 * (255 - a);
        let over = |c: u8| ((u32::from(c) * a + bg + 127) / 255).min(255) as u8;
        out.extend_from_slice(&[over(px[0]), over(px[1]), over(px[2])]);
    }
    out
}

////////////////////
// AnyImageSource //
////////////////////

/// Either backing implementation, picked by probing the file.
///
/// An enum rather than `Box<dyn ImageSource>` so the generic code downstream
/// stays monomorphic and the variant is visible to the caller.
#[derive(Debug)]
pub enum AnyImageSource {
    /// Fully decoded, held in memory.
    Whole(WholeImage),
    /// Windowed reads out of an OpenSlide pyramid.
    Slide(Box<SlideImage>),
}

impl ImageSource for AnyImageSource {
    fn dims(&self) -> (u32, u32) {
        match self {
            Self::Whole(s) => s.dims(),
            Self::Slide(s) => s.dims(),
        }
    }

    fn scale(&self) -> f32 {
        match self {
            Self::Whole(s) => s.scale(),
            Self::Slide(s) => s.scale(),
        }
    }

    fn read_window(&self, window: ImageWindow) -> Result<RgbTile, BixverseErrors> {
        match self {
            Self::Whole(s) => s.read_window(window),
            Self::Slide(s) => s.read_window(window),
        }
    }
}

/// Open a file by probing it, not by looking at the extension.
///
/// OpenSlide gets first refusal via `detect_vendor`, which sniffs the actual
/// container. Only if it declines do we fall back to a full decode. Extensions
/// lie constantly in this domain: plenty of `.tif` files are pyramidal slides
/// and plenty are not, and the bundled CytAssist `.tiff` is one of the latter.
///
/// ### Params
///
/// * `path` - File to open.
/// * `fallback_scale` - Scale to record if the file turns out to be a plain
///   image, where the scale cannot be recovered from the file itself. Ignored
///   on the slide path, which derives its scale from the pyramid.
/// * `working_scale` - Target scale for the slide path, used to choose the
///   pyramid level. Ignored on the whole-image path.
///
/// ### Returns
///
/// The opened source, or [`BixverseErrors::ImageSourceUnrecognised`] if neither
/// backend can read the file.
pub fn open_image_source<P: AsRef<Path>>(
    path: P,
    fallback_scale: f32,
    working_scale: f32,
) -> Result<AnyImageSource, BixverseErrors> {
    let path = path.as_ref();

    if SlideImage::is_slide(path) {
        return Ok(AnyImageSource::Slide(Box::new(SlideImage::new(
            path,
            working_scale,
        )?)));
    }

    match WholeImage::new(path, fallback_scale) {
        Ok(img) => Ok(AnyImageSource::Whole(img)),
        Err(_) => Err(BixverseErrors::ImageSourceUnrecognised {
            path: path.display().to_string(),
        }),
    }
}

//////////////
// Metadata //
//////////////

/// What a file says about itself before anything gets read out of it.
///
/// A plain image has no pyramid, so it reports one level, a downsample of 1.0
/// and no vendor. That keeps the shape identical across both backends and
/// saves the caller a branch.
#[derive(Clone, Debug)]
pub struct ImageMetadata {
    /// Width of level 0, in pixels.
    pub width: u32,
    /// Height of level 0, in pixels.
    pub height: u32,
    /// Number of pyramid levels. Always 1 for a plain image.
    pub n_levels: usize,
    /// Per-level `(width, height)`, level 0 first.
    pub level_dims: Vec<(u32, u32)>,
    /// Per-level downsample relative to level 0.
    pub downsamples: Vec<f64>,
    /// OpenSlide vendor string, `None` for a plain image.
    pub vendor: Option<String>,
}

/// Describe an image file without decoding it.
///
/// Probes with OpenSlide first, same rule as [`open_image_source`]. The plain
/// image path reads the header only, so this stays cheap on a 3 GB TIFF that
/// [`WholeImage::new`] would happily pull into memory.
///
/// ### Params
///
/// * `path` - File to describe.
///
/// ### Returns
///
/// The metadata, or [`BixverseErrors::ImageSourceUnrecognised`] if neither
/// backend can read the file.
pub fn image_metadata<P: AsRef<Path>>(path: P) -> Result<ImageMetadata, BixverseErrors> {
    let path = path.as_ref();

    if SlideImage::is_slide(path) {
        let meta = SlideImage::metadata(path)?;
        let (width, height) = meta.level_dims[0];
        return Ok(ImageMetadata {
            width,
            height,
            n_levels: meta.level_dims.len(),
            level_dims: meta.level_dims,
            downsamples: meta.downsamples,
            vendor: Some(meta.vendor),
        });
    }

    let (width, height) =
        image::image_dimensions(path).map_err(|_| BixverseErrors::ImageSourceUnrecognised {
            path: path.display().to_string(),
        })?;

    Ok(ImageMetadata {
        width,
        height,
        n_levels: 1,
        level_dims: vec![(width, height)],
        downsamples: vec![1.0],
        vendor: None,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A 4x3 image whose pixel `(x, y)` is `[x * 10, y * 10, 0]`.
    fn ramp() -> WholeImage {
        let mut data = Vec::with_capacity(4 * 3 * 3);
        for y in 0..3u8 {
            for x in 0..4u8 {
                data.extend_from_slice(&[x * 10, y * 10, 0]);
            }
        }
        WholeImage::from_rgb8(data, 4, 3, 1.0)
    }

    #[test]
    fn test_whole_image_full_window() {
        let img = ramp();
        let tile = img.read_window(ImageWindow::new(0, 0, 4, 3)).unwrap();
        assert_eq!(tile.width, 4);
        assert_eq!(tile.height, 3);
        assert_eq!(tile.n_pixels(), 12);
        assert_eq!(&tile.data[0..3], &[0, 0, 0]);
        assert_eq!(&tile.data[9..12], &[30, 0, 0]);
    }

    #[test]
    fn test_whole_image_interior_crop() {
        let img = ramp();
        let tile = img.read_window(ImageWindow::new(1, 1, 2, 2)).unwrap();
        assert_eq!((tile.width, tile.height), (2, 2));
        // Rows y = 1, 2 and columns x = 1, 2.
        assert_eq!(tile.data, vec![10, 10, 0, 20, 10, 0, 10, 20, 0, 20, 20, 0]);
    }

    #[test]
    fn test_whole_image_window_clamped_at_edge() {
        let img = ramp();
        let tile = img.read_window(ImageWindow::new(3, 2, 8, 8)).unwrap();
        assert_eq!((tile.width, tile.height), (1, 1));
        assert_eq!(tile.data, vec![30, 20, 0]);
    }

    #[test]
    fn test_whole_image_window_entirely_outside_errors() {
        let img = ramp();
        let err = img.read_window(ImageWindow::new(10, 10, 2, 2));
        assert!(matches!(
            err,
            Err(BixverseErrors::ImageWindowOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_rgba_transparency_composites_onto_white() {
        // `into_rgb8` drops alpha, so a fully transparent black pixel came out
        // as black, i.e. maximum optical density.
        let rgba = image::RgbaImage::from_raw(
            3,
            1,
            vec![
                0, 0, 0, 0, // transparent black -> white
                0, 0, 0, 255, // opaque black stays black
                255, 0, 0, 128, // half-transparent red -> pink
            ],
        )
        .unwrap();
        let rgb = rgba_onto_white(&rgba);
        assert_eq!(&rgb[0..3], &[255, 255, 255]);
        assert_eq!(&rgb[3..6], &[0, 0, 0]);
        assert_eq!(rgb[6], 255);
        assert!(rgb[7] > 120 && rgb[7] < 135, "{}", rgb[7]);
    }

    #[test]
    fn test_scale_is_reported_verbatim() {
        let img = WholeImage::from_rgb8(vec![0; 3], 1, 1, 0.056_121_446);
        assert_relative_eq!(img.scale(), 0.056_121_446, epsilon = 1e-9);
    }
}
