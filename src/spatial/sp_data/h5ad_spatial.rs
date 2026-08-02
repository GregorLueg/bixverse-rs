//! Reads the spatial extras out of an h5ad file.
//!
//! `obsm/spatial` and the `uns/spatial` group, and nothing else. Counts, `obs`
//! and `var` are the single cell reader's job and stay there.

use hdf5::types::{FloatSize, IntSize, TypeDescriptor};
use hdf5::{Dataset, File, Group};
use std::path::Path;

use crate::prelude::*;
use crate::spatial::sp_data::orientation::{
    OrientationCall, OrientationEvidence, SpatialOrientation, orientation_from_image_frame,
    orientation_from_obs_pixels, orientation_from_tissue_mask, tissue_mask,
};

/// The exact `obsm` key holding the coordinates.
///
/// Keyed exactly on purpose. The Vanderbilt files also carry `spatial_trim` and
/// `image_means`, so a fuzzy match on "spatial" picks up a cropped copy of the
/// coordinates instead of the ones the scale factors belong to.
const OBSM_SPATIAL_KEY: &str = "spatial";

/// `obs` column holding the full-res pixel row, when the file kept it.
const OBS_PXL_ROW: &str = "pxl_row_in_fullres";

/// `obs` column holding the full-res pixel column, when the file kept it.
const OBS_PXL_COL: &str = "pxl_col_in_fullres";

/// `obs` column holding the lattice row. Needed by the hex and square layouts.
const OBS_ARRAY_ROW: &str = "array_row";

/// `obs` column holding the lattice column.
const OBS_ARRAY_COL: &str = "array_col";

//////////////
// Structs //
//////////////

/// One image registered under `uns/spatial/<lib>/images`.
#[derive(Clone, Debug)]
pub struct SpatialImageEntry {
    /// Key it sits under, e.g. `lowres`. An open set: the Vanderbilt files add
    /// `hires_trim`.
    pub key: String,
    /// Height in pixels.
    pub height: usize,
    /// Width in pixels.
    pub width: usize,
}

/// Everything the spatial extras of one h5ad amount to.
#[derive(Clone, Debug)]
pub struct SpatialH5adData {
    /// Per-spot coordinates in full-res pixels, already reordered to the
    /// `(x, y)` contract. One entry per row of `obsm/spatial`, in file order.
    pub coordinates: Vec<(f64, f64)>,
    /// Which column of `obsm/spatial` was taken as `x`.
    pub orientation: SpatialOrientation,
    /// What settled the orientation.
    pub evidence: OrientationEvidence,
    /// Every key under `obsm`, so the caller can see what else was in there.
    pub obsm_keys: Vec<String>,
    /// The `uns/spatial` library identifier the rest of this refers to. `None`
    /// when the file carries no `uns/spatial` group at all.
    pub library_id: Option<String>,
    /// Every library identifier found, when there was more than one.
    pub library_ids: Vec<String>,
    /// Scale factors as `(key, value)` pairs, passed through untouched. The key
    /// set is open: Vanderbilt adds `tissue_hires_trim_scalef` to the standard
    /// four.
    pub scale_factors: Vec<(String, f64)>,
    /// Images present under the chosen library, with their pixel dimensions. A
    /// scale factor in `scale_factors` does **not** imply an entry here: every
    /// Li et al file ships a `tissue_hires_scalef` and no hires image.
    pub images: Vec<SpatialImageEntry>,
    /// Keys under `uns/spatial/<lib>/metadata`, which is optional and absent
    /// from the whole Li et al collection.
    pub metadata_keys: Vec<String>,
    /// Whether `obs` carries `array_row` and `array_col`. The hex and square
    /// graph layouts need them; kNN and radius do not.
    pub has_array_indices: bool,
    /// Whether `obs` carries the `pxl_row_in_fullres` / `pxl_col_in_fullres`
    /// pair, i.e. whether the orientation could be settled outright.
    pub has_pixel_columns: bool,
}

impl SpatialH5adData {
    /// Number of spots read.
    pub fn n_spots(&self) -> usize {
        self.coordinates.len()
    }
}

/// Knobs for [`read_spatial_h5ad`].
#[derive(Clone, Debug)]
pub struct SpatialH5adParams {
    /// Which `uns/spatial` library to read when the file carries several.
    /// `None` takes the only one, and errors when there is a choice to make.
    pub library_id: Option<String>,
    /// Column order to fall back on when nothing in the file settles it.
    /// Defaults to `(x, y)`, which is what `scanpy.read_visium` produces and
    /// what all 236 survey files hold.
    pub assumed_orientation: SpatialOrientation,
}

impl Default for SpatialH5adParams {
    fn default() -> Self {
        Self {
            library_id: None,
            assumed_orientation: SpatialOrientation::Xy,
        }
    }
}

impl SpatialH5adParams {
    /// Constructor.
    ///
    /// ### Params
    ///
    /// * `library_id` - `uns/spatial` library to read, or `None` for the only
    ///   one present.
    /// * `assumed_orientation` - Fallback column order.
    pub fn new(library_id: Option<String>, assumed_orientation: SpatialOrientation) -> Self {
        Self {
            library_id,
            assumed_orientation,
        }
    }
}

/////////////
// Helpers //
/////////////

/// Read a numeric HDF5 dataset into `f64` whatever it is stored as.
///
/// `obsm/spatial` is `int64` in all 236 survey files, the scale factors mix
/// `int64` and `float64` within one group, and nothing in the spec stops a
/// third writer from using `float32`. Reading through the descriptor keeps that
/// from being a per-collection surprise.
///
/// ### Params
///
/// * `ds` - The dataset to read.
///
/// ### Returns
///
/// The values in file order, or [`BixverseErrors::H5UnexpectedNumericType`]
/// for a non-numeric dataset.
fn read_numeric_1d(ds: &Dataset) -> Result<Vec<f64>, BixverseErrors> {
    let values = match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(FloatSize::U8) => ds.read_raw::<f64>()?,
        TypeDescriptor::Float(FloatSize::U4) => {
            ds.read_raw::<f32>()?.into_iter().map(f64::from).collect()
        }
        TypeDescriptor::Integer(IntSize::U8) => ds
            .read_raw::<i64>()?
            .into_iter()
            .map(|v| v as f64)
            .collect(),
        TypeDescriptor::Integer(IntSize::U4) => {
            ds.read_raw::<i32>()?.into_iter().map(f64::from).collect()
        }
        TypeDescriptor::Integer(IntSize::U2) => {
            ds.read_raw::<i16>()?.into_iter().map(f64::from).collect()
        }
        TypeDescriptor::Integer(IntSize::U1) => {
            ds.read_raw::<i8>()?.into_iter().map(f64::from).collect()
        }
        TypeDescriptor::Unsigned(IntSize::U8) => ds
            .read_raw::<u64>()?
            .into_iter()
            .map(|v| v as f64)
            .collect(),
        TypeDescriptor::Unsigned(IntSize::U4) => {
            ds.read_raw::<u32>()?.into_iter().map(f64::from).collect()
        }
        TypeDescriptor::Unsigned(IntSize::U2) => {
            ds.read_raw::<u16>()?.into_iter().map(f64::from).collect()
        }
        TypeDescriptor::Unsigned(IntSize::U1) => {
            ds.read_raw::<u8>()?.into_iter().map(f64::from).collect()
        }
        other => {
            return Err(BixverseErrors::H5UnexpectedNumericType {
                path: ds.name(),
                dtype: format!("{other:?}"),
            });
        }
    };

    Ok(values)
}

/// Read an `obs` column into `f64`, or `None` when it is absent.
///
/// Absent is the normal case for half of these columns, so a missing dataset is
/// not an error. A present but non-numeric one is: a categorical `array_row`
/// would otherwise be read as "not there" and quietly cost the caller the
/// lattice layouts.
///
/// ### Params
///
/// * `file` - The open h5ad.
/// * `name` - Column name under `obs`.
///
/// ### Returns
///
/// The column, or `None` when `obs/<name>` does not exist as a dataset.
fn read_obs_numeric(file: &File, name: &str) -> Result<Option<Vec<f64>>, BixverseErrors> {
    match file.dataset(&format!("obs/{name}")) {
        Ok(ds) => read_numeric_1d(&ds).map(Some),
        Err(_) => Ok(None),
    }
}

/// Read `obsm/spatial` into its two columns.
///
/// ### Params
///
/// * `file` - The open h5ad.
///
/// ### Returns
///
/// `(col0, col1)`, or an error when the array is missing or not N x 2.
fn read_obsm_spatial(file: &File) -> Result<(Vec<f64>, Vec<f64>), BixverseErrors> {
    let path = format!("obsm/{OBSM_SPATIAL_KEY}");
    let ds = file
        .dataset(&path)
        .map_err(|_| BixverseErrors::SpatialObsmMissing { path: path.clone() })?;

    let shape = ds.shape();
    if shape.len() != 2 || shape[1] < 2 {
        return Err(BixverseErrors::SpatialObsmShape {
            path,
            shape: format!("{shape:?}"),
        });
    }

    // Row-major, so the two columns interleave. Anything past column 1 is
    // dropped: a 3-D `obsm/spatial` is a z stack this container has no notion of.
    let n_cols = shape[1];
    let flat = read_numeric_1d(&ds)?;
    if flat.len() != shape[0] * n_cols {
        return Err(BixverseErrors::SpatialObsmShape {
            path,
            shape: format!("{shape:?} but {} values", flat.len()),
        });
    }

    let mut col0 = Vec::with_capacity(shape[0]);
    let mut col1 = Vec::with_capacity(shape[0]);
    for row in flat.chunks_exact(n_cols) {
        col0.push(row[0]);
        col1.push(row[1]);
    }

    Ok((col0, col1))
}

/// Pick the `uns/spatial` library to read.
///
/// ### Params
///
/// * `group` - The `uns/spatial` group.
/// * `requested` - Library the caller asked for, if any.
///
/// ### Returns
///
/// `(chosen, all)`, or an error when the request misses or the choice is
/// ambiguous.
fn resolve_library(
    group: &Group,
    requested: &Option<String>,
) -> Result<(String, Vec<String>), BixverseErrors> {
    let mut available: Vec<String> = group.member_names()?;
    available.sort();

    if let Some(want) = requested {
        if !available.iter().any(|k| k == want) {
            return Err(BixverseErrors::SpatialLibraryNotFound {
                library_id: want.clone(),
                available: available.join(", "),
            });
        }
        return Ok((want.clone(), available));
    }

    match available.len() {
        0 => Err(BixverseErrors::SpatialLibraryNotFound {
            library_id: "<any>".to_string(),
            available: "none".to_string(),
        }),
        1 => Ok((available[0].clone(), available)),
        _ => Err(BixverseErrors::SpatialLibraryAmbiguous {
            available: available.join(", "),
        }),
    }
}

/// Read the scale factors of one library.
///
/// Each key is its own scalar dataset rather than a JSON blob, and the key set
/// is open, so everything present is passed through.
///
/// ### Params
///
/// * `lib_group` - The `uns/spatial/<lib>` group.
///
/// ### Returns
///
/// `(key, value)` pairs, sorted by key. Empty when the group has no
/// `scalefactors` member.
fn read_scale_factors(lib_group: &Group) -> Result<Vec<(String, f64)>, BixverseErrors> {
    let Ok(sf_group) = lib_group.group("scalefactors") else {
        return Ok(Vec::new());
    };

    let mut keys = sf_group.member_names()?;
    keys.sort();

    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        let ds = sf_group.dataset(&key)?;
        let values = read_numeric_1d(&ds)?;
        // Scalar datasets read back as a single value. Anything else is not a
        // scale factor and gets skipped rather than mangled into one.
        if values.len() == 1 {
            out.push((key, values[0]));
        }
    }

    Ok(out)
}

/// List the images of one library with their pixel dimensions.
///
/// The pixel data itself is never read: these files carry `float32` H x W x 3
/// arrays and the dimensions are all the reader needs.
///
/// ### Params
///
/// * `lib_group` - The `uns/spatial/<lib>` group.
///
/// ### Returns
///
/// One [`SpatialImageEntry`] per image, sorted by key.
fn read_image_entries(lib_group: &Group) -> Result<Vec<SpatialImageEntry>, BixverseErrors> {
    let Ok(img_group) = lib_group.group("images") else {
        return Ok(Vec::new());
    };

    let mut keys = img_group.member_names()?;
    keys.sort();

    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        let Ok(ds) = img_group.dataset(&key) else {
            continue;
        };
        let shape = ds.shape();
        if shape.len() >= 2 {
            out.push(SpatialImageEntry {
                key,
                height: shape[0],
                width: shape[1],
            });
        }
    }

    Ok(out)
}

/// Find an image that pairs with a usable scale factor.
///
/// `lowres` first: it is the one the orientation tests read whole, and at
/// 600 px on its longest side it costs a few megabytes rather than fifty.
///
/// ### Params
///
/// * `images` - Images of the chosen library.
/// * `scale_factors` - Scale factors of the chosen library.
///
/// ### Returns
///
/// The image and its scale factor, or `None` when nothing pairs up.
fn pick_reference_image<'a>(
    images: &'a [SpatialImageEntry],
    scale_factors: &[(String, f64)],
) -> Option<(&'a SpatialImageEntry, f64)> {
    let lookup = |key: &str| -> Option<f64> {
        scale_factors
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| *v)
            .filter(|v| v.is_finite() && *v > 0.0)
    };

    for (img_key, sf_key) in [
        ("lowres", "tissue_lowres_scalef"),
        ("hires", "tissue_hires_scalef"),
    ] {
        // A scale factor does not imply the matching image exists: every Li et
        // al file ships a `tissue_hires_scalef` and no hires image.
        if let Some(entry) = images.iter().find(|e| e.key == img_key)
            && let Some(scalef) = lookup(sf_key)
        {
            return Some((entry, scalef));
        }
    }

    None
}

/// Largest image the mask test will decode, in pixels.
///
/// A `lowres` frame is well under a megapixel. This only bites when a library
/// ships no `lowres` and its `hires` is enormous, where reading H x W x 3
/// `float32` would cost more than the answer is worth.
const MAX_MASK_PIXELS: usize = 25_000_000;

/// Read one image out of `uns/spatial/<lib>/images` as greyscale on 0..1.
///
/// ### Params
///
/// * `lib_group` - The `uns/spatial/<lib>` group.
/// * `entry` - Which image to read.
///
/// ### Returns
///
/// The row-major greyscale image, or `None` when it is too large, not
/// three-dimensional, or unreadable.
fn read_grey_image(lib_group: &Group, entry: &SpatialImageEntry) -> Option<Vec<f64>> {
    if entry.height.saturating_mul(entry.width) > MAX_MASK_PIXELS {
        return None;
    }

    let ds = lib_group.group("images").ok()?.dataset(&entry.key).ok()?;
    let shape = ds.shape();
    if shape.len() != 3 {
        return None;
    }
    let channels = shape[2];
    if channels == 0 {
        return None;
    }

    let flat = read_numeric_1d(&ds).ok()?;
    if flat.len() != entry.height * entry.width * channels {
        return None;
    }

    // These arrive as `float32` on 0..1 in both collections, but an 8-bit
    // image round-trips through h5py as 0..255 just as easily.
    let max = flat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let divisor = if max > 1.5 { 255.0 } else { 1.0 };

    Some(
        flat.chunks_exact(channels)
            .map(|px| px.iter().sum::<f64>() / (channels as f64 * divisor))
            .collect(),
    )
}

/// Settle the column order of `obsm/spatial`.
///
/// Ranked by how directly the evidence measures the one thing the order
/// affects, which is whether the coordinates land on the histology image. The
/// tissue mask tests that outright, the frame bound is a weaker form of the
/// same idea, and the `obs` column names come last because `scanpy.read_visium`
/// swaps them relative to Space Ranger.
///
/// ### Params
///
/// * `col0` - Column 0 of `obsm/spatial`.
/// * `col1` - Column 1.
/// * `reference` - The image to test against with its scale factor, when one is
///   available, together with its greyscale pixels.
/// * `pxl_row` - `obs/pxl_row_in_fullres`, when present.
/// * `pxl_col` - `obs/pxl_col_in_fullres`, when present.
/// * `assumed` - Fallback column order.
///
/// ### Returns
///
/// The call together with what produced it.
fn resolve_orientation(
    col0: &[f64],
    col1: &[f64],
    reference: Option<(&SpatialImageEntry, f64, Option<Vec<f64>>)>,
    pxl_row: Option<&[f64]>,
    pxl_col: Option<&[f64]>,
    assumed: SpatialOrientation,
) -> OrientationCall {
    if let Some((entry, scalef, grey)) = reference {
        if let Some(grey) = grey
            && let Some(mask) = tissue_mask(&grey, entry.height, entry.width)
            && let Some(orientation) =
                orientation_from_tissue_mask(col0, col1, &mask, entry.height, entry.width, scalef)
        {
            return OrientationCall {
                orientation,
                evidence: OrientationEvidence::ImageTissue,
            };
        }

        let max0 = col0.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let max1 = col1.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if let Some(orientation) = orientation_from_image_frame(
            max0,
            max1,
            entry.height as f64 / scalef,
            entry.width as f64 / scalef,
        ) {
            return OrientationCall {
                orientation,
                evidence: OrientationEvidence::ImageFrame,
            };
        }
    }

    if let (Some(row), Some(col)) = (pxl_row, pxl_col)
        && let Some(orientation) = orientation_from_obs_pixels(col0, col1, row, col)
    {
        return OrientationCall {
            orientation,
            evidence: OrientationEvidence::ObsPixelColumns,
        };
    }

    OrientationCall {
        orientation: assumed,
        evidence: OrientationEvidence::Assumed,
    }
}

////////////
// Reader //
////////////

/// Read the spatial extras out of an h5ad.
///
/// Touches `obsm`, four optional `obs` columns and the `uns/spatial` group.
/// The counts, the full `obs` table and `var` are the single cell reader's job.
///
/// The returned coordinates are already in the `(x, y)` contract, so the caller
/// never sees the raw column order. What produced that order is reported in
/// [`SpatialH5adData::evidence`] and is worth surfacing: on
/// [`OrientationEvidence::Assumed`] nothing in the file confirmed it.
///
/// ### Params
///
/// * `h5_path` - Path to the `.h5ad` file.
/// * `params` - Optional [`SpatialH5adParams`]; `None` takes the defaults.
///
/// ### Returns
///
/// The coordinates, the orientation call and whatever else survived in
/// `uns/spatial`.
pub fn read_spatial_h5ad<P: AsRef<Path>>(
    h5_path: P,
    params: Option<SpatialH5adParams>,
) -> Result<SpatialH5adData, BixverseErrors> {
    let params = params.unwrap_or_default();
    let file = File::open(h5_path.as_ref())?;

    let (col0, col1) = read_obsm_spatial(&file)?;
    let n_spots = col0.len();

    let mut obsm_keys = file
        .group("obsm")
        .and_then(|g| g.member_names())
        .unwrap_or_default();
    obsm_keys.sort();

    // A column that does not line up with `obsm/spatial` is not this file's
    // coordinate pair and is dropped rather than compared against.
    let usable = |v: Option<Vec<f64>>| v.filter(|c| c.len() == n_spots);
    let pxl_row = usable(read_obs_numeric(&file, OBS_PXL_ROW)?);
    let pxl_col = usable(read_obs_numeric(&file, OBS_PXL_COL)?);
    let array_row = usable(read_obs_numeric(&file, OBS_ARRAY_ROW)?);
    let array_col = usable(read_obs_numeric(&file, OBS_ARRAY_COL)?);

    let mut lib_group = None;
    let (library_id, library_ids, scale_factors, images, metadata_keys) =
        match file.group("uns/spatial") {
            Ok(group) => {
                let (chosen, all) = resolve_library(&group, &params.library_id)?;
                let chosen_group = group.group(&chosen)?;
                let scale_factors = read_scale_factors(&chosen_group)?;
                let images = read_image_entries(&chosen_group)?;
                let mut metadata_keys = chosen_group
                    .group("metadata")
                    .and_then(|g| g.member_names())
                    .unwrap_or_default();
                metadata_keys.sort();
                lib_group = Some(chosen_group);
                (Some(chosen), all, scale_factors, images, metadata_keys)
            }
            Err(_) => (None, Vec::new(), Vec::new(), Vec::new(), Vec::new()),
        };

    let reference = lib_group.as_ref().and_then(|group| {
        pick_reference_image(&images, &scale_factors)
            .map(|(entry, scalef)| (entry, scalef, read_grey_image(group, entry)))
    });

    let call = resolve_orientation(
        &col0,
        &col1,
        reference,
        pxl_row.as_deref(),
        pxl_col.as_deref(),
        params.assumed_orientation,
    );

    let coordinates: Vec<(f64, f64)> = col0
        .iter()
        .zip(col1.iter())
        .map(|(&a, &b)| call.orientation.to_xy(a, b))
        .collect();

    for (index, &(x, y)) in coordinates.iter().enumerate() {
        if !x.is_finite() || !y.is_finite() {
            return Err(BixverseErrors::SpatialNonFiniteCoord { index });
        }
    }

    Ok(SpatialH5adData {
        coordinates,
        orientation: call.orientation,
        evidence: call.evidence,
        obsm_keys,
        library_id,
        library_ids,
        scale_factors,
        images,
        metadata_keys,
        has_array_indices: array_row.is_some() && array_col.is_some(),
        has_pixel_columns: pxl_row.is_some() && pxl_col.is_some(),
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(key: &str, height: usize, width: usize) -> SpatialImageEntry {
        SpatialImageEntry {
            key: key.to_string(),
            height,
            width,
        }
    }

    #[test]
    fn test_resolve_orientation_lets_the_image_overrule_the_obs_labels() {
        // The Li et al case in miniature. Column 0 equals `pxl_row_in_fullres`,
        // so the labels say `Yx`, but column 0 runs past the bottom of the
        // slide so the frame refutes it. The image wins, because scanpy swaps
        // those two names on the way in.
        let col0 = [100.0, 9055.0];
        let col1 = [50.0, 6009.0];
        let img = entry("lowres", 767, 1122);
        let call = resolve_orientation(
            &col0,
            &col1,
            Some((&img, 0.1, None)),
            Some(&col0),
            Some(&col1),
            SpatialOrientation::Xy,
        );
        assert_eq!(call.orientation, SpatialOrientation::Xy);
        assert_eq!(call.evidence, OrientationEvidence::ImageFrame);
    }

    #[test]
    fn test_resolve_orientation_uses_the_labels_when_no_image_ships() {
        let pxl_row = [10.0, 4000.0];
        let pxl_col = [20.0, 30.0];
        let call = resolve_orientation(
            &pxl_row,
            &pxl_col,
            None,
            Some(&pxl_row),
            Some(&pxl_col),
            SpatialOrientation::Xy,
        );
        assert_eq!(call.orientation, SpatialOrientation::Yx);
        assert_eq!(call.evidence, OrientationEvidence::ObsPixelColumns);
    }

    #[test]
    fn test_resolve_orientation_admits_when_it_is_guessing() {
        let col0 = [1.0, 2.0];
        let col1 = [3.0, 4.0];
        let call = resolve_orientation(&col0, &col1, None, None, None, SpatialOrientation::Yx);
        assert_eq!(call.orientation, SpatialOrientation::Yx);
        assert_eq!(call.evidence, OrientationEvidence::Assumed);
    }

    #[test]
    fn test_pick_reference_image_prefers_lowres() {
        let images = vec![entry("hires", 2000, 1000), entry("lowres", 600, 300)];
        let sf = vec![
            ("tissue_hires_scalef".to_string(), 0.2),
            ("tissue_lowres_scalef".to_string(), 0.06),
        ];
        let (chosen, scalef) = pick_reference_image(&images, &sf).unwrap();
        assert_eq!(chosen.key, "lowres");
        assert!((scalef - 0.06).abs() < 1e-12);
    }

    #[test]
    fn test_pick_reference_image_needs_the_matching_image() {
        // Every Li et al file ships a `tissue_hires_scalef` and no hires image.
        let images = vec![entry("lowres", 600, 300)];
        let sf = vec![("tissue_hires_scalef".to_string(), 1.0)];
        assert!(pick_reference_image(&images, &sf).is_none());
    }

    #[test]
    fn test_pick_reference_image_rejects_a_zero_scale_factor() {
        let images = vec![entry("lowres", 600, 300)];
        let sf = vec![("tissue_lowres_scalef".to_string(), 0.0)];
        assert!(pick_reference_image(&images, &sf).is_none());
    }
}
