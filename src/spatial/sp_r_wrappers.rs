//! Contains R-specific functions for spatial data processing that need the
//! extendr interface.

use extendr_api::*;
use std::collections::HashMap;

use crate::spatial::sp_analysis::nhood_enrichment::NhoodEnrichmentParams;
use crate::spatial::sp_graph::{SpatialGraphLayout, SpatialGraphParams, SpatialWeighting};
use crate::spatial::sp_processing::sparkx::{SparkXKernel, SparkXParams};
use crate::spatial::sp_processing::svg_utils::{SpatialSvgParams, parse_assay_type};

#[cfg(feature = "spatial-image")]
use crate::prelude::IntoExtendrErr;
#[cfg(feature = "spatial-image")]
use crate::spatial::sp_image::SpatialImageParams;
#[cfg(feature = "spatial-image")]
use crate::spatial::sp_image::colour::StainMatrix;

/////////////
// Helpers //
/////////////

/// Pull an integer vector out of an R list entry.
///
/// `NULL` and a missing key both come back as `None`, which is what lets the
/// caller fall back to a default rather than erroring on an absent optional.
///
/// ### Params
///
/// * `map` - The list, already converted to a map.
/// * `key` - Which element to read.
///
/// ### Returns
///
/// The values as `i32`, or `None`.
fn opt_int_vec(map: &HashMap<&str, Robj>, key: &str) -> Option<Vec<i32>> {
    map.get(key)
        .filter(|v| !v.is_null())
        .and_then(|v| v.as_integer_vector())
}

/// Pull a real vector out of an R list entry. See [`opt_int_vec`].
///
/// ### Params
///
/// * `map` - The list, already converted to a map.
/// * `key` - Which element to read.
///
/// ### Returns
///
/// The values as `f64`, or `None`.
fn opt_real_vec(map: &HashMap<&str, Robj>, key: &str) -> Option<Vec<f64>> {
    map.get(key)
        .filter(|v| !v.is_null())
        .and_then(|v| v.as_real_vector())
}

/// Pull a scalar double out of an R list entry. See [`opt_int_vec`].
///
/// ### Params
///
/// * `map` - The list, already converted to a map.
/// * `key` - Which element to read.
///
/// ### Returns
///
/// The value, or `None`.
fn opt_real(map: &HashMap<&str, Robj>, key: &str) -> Option<f64> {
    map.get(key)
        .filter(|v| !v.is_null())
        .and_then(|v| v.as_real())
}

////////////
// SparkX //
////////////

impl SparkXKernel {
    /// Generate a [SparkXKernel] from an R list
    ///
    /// ### Params
    ///
    /// * `r_list` - The list to convert into a Kernel
    ///
    /// ### Returns
    ///
    /// [SparkXKernel] if successful
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let map: HashMap<&str, Robj> = r_list.try_into()?;

        let bandwidth = map
            .get("bandwidth")
            .and_then(|v| v.as_real())
            .ok_or_else(|| {
                Error::Other(
                    "The provided bandwidth parameter could not be extracted correctly.".into(),
                )
            })? as f32;

        let kernel = map
            .get("kernel")
            .and_then(|x| x.as_str())
            .and_then(|x| match x {
                "gaussian" => Some(SparkXKernel::Gaussian { bandwidth }),
                "cosine" => Some(SparkXKernel::Cosine { bandwidth }),
                _ => None,
            })
            .ok_or_else(|| {
                Error::Other(
                    "The provided kernel parameter could not be extracted correctly.".into(),
                )
            })?;

        Ok(kernel)
    }
}

impl SparkXParams {
    /// Generate a [SparkXParams] from an R list
    ///
    /// ### Params
    ///
    /// * `r_list` - The list to convert into a Kernel
    ///
    /// ### Returns
    ///
    /// [SparkXParams] if successful
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list.try_into()?;
        let defaults = Self::default();

        let kernels = params
            .get("kernels")
            .and_then(|v| if v.is_null() { None } else { v.as_list() })
            .map(|list| {
                list.into_iter()
                    .map(|(_, robj)| {
                        List::try_from(robj)
                            .map_err(|e| Error::Other(e.into()))
                            .and_then(SparkXKernel::from_r_list)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();

        let n_landmarks = params
            .get("n_landmarks")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.n_landmarks);

        let bandwidth_subsample = params
            .get("bandwidth_subsample")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.bandwidth_subsample);

        Ok(Self {
            kernels,
            n_landmarks,
            bandwidth_subsample,
        })
    }
}

/////////////////////////////
// Nhood enrichment params //
/////////////////////////////

impl NhoodEnrichmentParams {
    /// Generate a [NhoodEnrichmentParams] from an R list
    ///
    /// ### Params
    ///
    /// * `r_list` - The list to convert into a Kernel
    ///
    /// ### Returns
    ///
    /// [NhoodEnrichmentParams] if successful
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list.try_into()?;
        let defaults = Self::default();

        let n_perm = params
            .get("n_perm")
            .and_then(|x| x.as_integer())
            .map(|x| x as usize)
            .unwrap_or(defaults.n_perm);

        let symmetrise = params
            .get("symmetrise")
            .and_then(|x| x.as_bool())
            .unwrap_or(defaults.symmetrise);

        Ok(Self { n_perm, symmetrise })
    }
}

//////////////////
// Spatial graph //
//////////////////

impl SpatialGraphParams {
    /// Generate a [SpatialGraphParams] from an R list.
    ///
    /// The list is flat rather than nested, because the R side already
    /// validates it as one parameter bundle. `layout` and `weighting` are
    /// strings and the fields each variant needs sit alongside them:
    ///
    /// * `layout = "hex"` needs `array_row` and `array_col`.
    /// * `layout = "square"` needs those plus `connectivity` (4 or 8).
    /// * `layout = "knn"` needs `k`.
    /// * `layout = "radius"` needs `radius`.
    /// * `weighting = "inverse_distance"` needs `power`.
    /// * `weighting = "gaussian"` takes `bandwidth`, `NULL` to auto-resolve.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list to convert.
    ///
    /// ### Returns
    ///
    /// [SpatialGraphParams] if successful.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list.try_into()?;
        let defaults = Self::default();

        let lattice = |key: &str| -> Result<Vec<i32>> {
            opt_int_vec(&params, key).ok_or_else(|| {
                Error::Other(format!(
                    "The spatial graph layout needs `{key}` as integers."
                ))
            })
        };

        let layout = match params.get("layout").and_then(|x| x.as_str()) {
            Some("hex") => SpatialGraphLayout::HexLattice {
                array_row: lattice("array_row")?,
                array_col: lattice("array_col")?,
            },
            Some("square") => SpatialGraphLayout::SquareLattice {
                array_row: lattice("array_row")?,
                array_col: lattice("array_col")?,
                connectivity: params
                    .get("connectivity")
                    .and_then(|x| x.as_integer())
                    .ok_or_else(|| {
                        Error::Other("The square lattice layout needs `connectivity`.".into())
                    })? as u8,
            },
            Some("knn") => SpatialGraphLayout::Knn {
                k: params
                    .get("k")
                    .and_then(|x| x.as_integer())
                    .map(|x| x as usize)
                    .ok_or_else(|| Error::Other("The kNN layout needs `k`.".into()))?,
            },
            Some("radius") => SpatialGraphLayout::Radius {
                radius: opt_real(&params, "radius")
                    .ok_or_else(|| Error::Other("The radius layout needs `radius`.".into()))?
                    as f32,
            },
            _ => {
                return Err(Error::Other(
                    "`layout` must be one of 'hex', 'square', 'knn' or 'radius'.".into(),
                ));
            }
        };

        let weighting = match params.get("weighting").and_then(|x| x.as_str()) {
            Some("binary") => SpatialWeighting::Binary,
            Some("inverse_distance") => SpatialWeighting::InverseDistance {
                power: opt_real(&params, "power").ok_or_else(|| {
                    Error::Other("The inverse distance weighting needs `power`.".into())
                })? as f32,
            },
            // `None` here is meaningful: it asks for the median nearest
            // neighbour distance rather than being a missing value.
            Some("gaussian") => SpatialWeighting::Gaussian {
                bandwidth: opt_real(&params, "bandwidth").map(|x| x as f32),
            },
            _ => {
                return Err(Error::Other(
                    "`weighting` must be one of 'binary', 'inverse_distance' or 'gaussian'.".into(),
                ));
            }
        };

        let row_normalise = params
            .get("row_normalise")
            .and_then(|x| x.as_bool())
            .unwrap_or(defaults.row_normalise);

        Ok(Self {
            layout,
            weighting,
            row_normalise,
        })
    }
}

//////////////////////
// Spatial SVG params //
//////////////////////

impl SpatialSvgParams {
    /// Generate a [SpatialSvgParams] from an R list.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list to convert. Reads `assay`, one of `"raw"` or
    ///   `"norm"`.
    ///
    /// ### Returns
    ///
    /// [SpatialSvgParams] if successful.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list.try_into()?;
        let defaults = Self::default();

        let assay = match params.get("assay").and_then(|x| x.as_str()) {
            Some(s) => parse_assay_type(s).ok_or_else(|| {
                Error::Other(format!("`assay` must be 'raw' or 'norm', got '{s}'."))
            })?,
            None => defaults.assay,
        };

        Ok(Self { assay })
    }
}

///////////////////////////
// Spatial image params //
///////////////////////////

#[cfg(feature = "spatial-image")]
impl SpatialImageParams {
    /// Generate a [SpatialImageParams] from an R list.
    ///
    /// GLCM offsets arrive as two aligned integer vectors, `glcm_offsets_dy`
    /// and `glcm_offsets_dx`, rather than one interleaved vector. An
    /// interleaved one silently transposes the whole offset set when a caller
    /// gets the order wrong, and there is nothing downstream that would notice.
    ///
    /// The stain basis is given as `stain_haem` and `stain_eosin`, three
    /// doubles each. Both must be present or both absent; the residual stain
    /// is derived by [`StainMatrix::from_two_stains`].
    ///
    /// ### Params
    ///
    /// * `r_list` - The list to convert.
    ///
    /// ### Returns
    ///
    /// [SpatialImageParams] if successful.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list.try_into()?;
        let defaults = Self::default();

        let tile_scale = opt_real(&params, "tile_scale")
            .map(|x| x as f32)
            .unwrap_or(defaults.tile_scale);

        let glcm_levels = params
            .get("glcm_levels")
            .and_then(|x| x.as_integer())
            .map(|x| x as u8)
            .unwrap_or(defaults.glcm_levels);

        let glcm_offsets = match (
            opt_int_vec(&params, "glcm_offsets_dy"),
            opt_int_vec(&params, "glcm_offsets_dx"),
        ) {
            (Some(dy), Some(dx)) => {
                if dy.len() != dx.len() {
                    return Err(Error::Other(format!(
                        "`glcm_offsets_dy` ({}) and `glcm_offsets_dx` ({}) must be the same length.",
                        dy.len(),
                        dx.len()
                    )));
                }
                dy.into_iter()
                    .zip(dx)
                    .map(|(y, x)| (y as i8, x as i8))
                    .collect()
            }
            (None, None) => defaults.glcm_offsets,
            _ => {
                return Err(Error::Other(
                    "`glcm_offsets_dy` and `glcm_offsets_dx` must be given together.".into(),
                ));
            }
        };

        let stain_matrix = match (
            opt_real_vec(&params, "stain_haem"),
            opt_real_vec(&params, "stain_eosin"),
        ) {
            (Some(haem), Some(eosin)) => {
                if haem.len() != 3 || eosin.len() != 3 {
                    return Err(Error::Other(
                        "`stain_haem` and `stain_eosin` must each hold three values.".into(),
                    ));
                }
                StainMatrix::from_two_stains(
                    [haem[0], haem[1], haem[2]],
                    [eosin[0], eosin[1], eosin[2]],
                )
                .to_extendr()?
            }
            (None, None) => defaults.stain_matrix,
            _ => {
                return Err(Error::Other(
                    "`stain_haem` and `stain_eosin` must be given together.".into(),
                ));
            }
        };

        Ok(Self {
            tile_scale,
            glcm_levels,
            glcm_offsets,
            stain_matrix,
        })
    }
}
