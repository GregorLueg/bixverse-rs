//! R wrappers over structures within the `graph` module

use extendr_api::*;
use std::collections::HashMap;

use crate::prelude::*;
use bixverse_rs::graph::knn_graph::{KnnSymmetrisation, SnnSimilarity, parse_snn_similarity};
use bixverse_rs::graph::leiden::{LeidenParams, LocalMoving, parse_local_moving};

//////////////////
// LeidenParams //
//////////////////

impl FromRList for LeidenParams {
    /// Parse the [LeidenParams] from an R list.
    ///
    /// Missing keys fall back to [LeidenParams::default]. Recognised keys:
    /// `resolution`, `beta`, `n_iterations` (negative: until stable),
    /// `min_gain`, `refine`, `seed` and `local_moving` (`"queue"` or
    /// `"coloured"`).
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list to parse
    ///
    /// ### Returns
    ///
    /// The [LeidenParams], or an error when a value is present but unusable.
    fn from_r_list(r_list: List) -> Result<Self> {
        let params = r_list_to_map(r_list)?;
        let defaults = Self::default();

        Ok(Self {
            resolution: r_list_real(&params, "resolution")?.unwrap_or(defaults.resolution),
            beta: r_list_real(&params, "beta")?.unwrap_or(defaults.beta),
            n_iterations: r_leiden_iterations(&params)?.unwrap_or(defaults.n_iterations),
            min_gain: r_list_real(&params, "min_gain")?.unwrap_or(defaults.min_gain),
            refine: params
                .get("refine")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.refine),
            seed: r_list_count_allow_zero(&params, "seed")?
                .map(|v| v as u64)
                .unwrap_or(defaults.seed),
            local_moving: r_local_moving(&params)?.unwrap_or(defaults.local_moving),
        })
    }
}

/// Read the optional `local_moving` key.
///
/// ### Params
///
/// * `params` - The flattened R list
///
/// ### Returns
///
/// The [LocalMoving] if the key is there, or an error for an unknown name
pub fn r_local_moving(params: &HashMap<&str, Robj>) -> Result<Option<LocalMoving>> {
    match params.get("local_moving").and_then(|v| v.as_str()) {
        None => Ok(None),
        Some(s) => parse_local_moving(s).map(Some).ok_or_else(|| {
            Error::Other(format!(
                "Unknown local_moving '{s}'; expected 'queue' or 'coloured'"
            ))
        }),
    }
}

/// Read the optional `n_iterations` key of the Leiden parameters.
///
/// As in igraph, a negative value means "until stable".
///
/// ### Params
///
/// * `params` - The flattened R list
///
/// ### Returns
///
/// `None` if the key is absent, `Some(None)` for a negative value,
/// `Some(Some(k))` for a positive whole number, or an error for 0 or a
/// fraction
pub fn r_leiden_iterations(params: &HashMap<&str, Robj>) -> Result<Option<Option<usize>>> {
    match r_list_real(params, "n_iterations")? {
        None => Ok(None),
        Some(v) if v < 0.0 => Ok(Some(None)),
        Some(v) if v >= 1.0 && v.fract() == 0.0 => Ok(Some(Some(v as usize))),
        Some(v) => Err(Error::Other(format!(
            "n_iterations must be a positive whole number, or negative to iterate until stable; got {v}"
        ))),
    }
}

///////////////////////
// KnnSymmetrisation //
///////////////////////

impl FromRMap for KnnSymmetrisation {
    /// Parse the [KnnSymmetrisation] from a flattened R list.
    ///
    /// `knn_sym` picks the variant: `"snn"`, `"umap"` or `"knn"`; missing
    /// gives [KnnSymmetrisation::default]. Per variant: `snn_similarity`
    /// (`"jaccard"` or `"rank"`), `pruning` and `rescue`; `local_connectivity`
    /// and `mix_ratio`; `binary`. Missing keys take the variant's default.
    ///
    /// ### Params
    ///
    /// * `params` - The flattened R list
    ///
    /// ### Returns
    ///
    /// The [KnnSymmetrisation], or an error for an unknown variant or
    /// similarity.
    fn from_r_map(params: &HashMap<&str, Robj>) -> Result<Self> {
        let Some(kind) = params.get("knn_sym").and_then(|v| v.as_str()) else {
            return Ok(Self::default());
        };
        match kind.to_lowercase().as_str() {
            "snn" => {
                let similarity = match params.get("snn_similarity").and_then(|v| v.as_str()) {
                    None => SnnSimilarity::default(),
                    Some(s) => parse_snn_similarity(s)
                        .ok_or_else(|| Error::Other(format!("Unknown snn_similarity '{s}'")))?,
                };
                let (prune, rescue) = match Self::default() {
                    Self::Snn { prune, rescue, .. } => (prune, rescue),
                    _ => (1.0 / 15.0, true),
                };
                Ok(Self::Snn {
                    similarity,
                    prune: r_list_real(params, "pruning")?.unwrap_or(prune),
                    rescue: params
                        .get("rescue")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(rescue),
                })
            }
            "umap" => Ok(Self::Umap {
                local_connectivity: r_list_real(params, "local_connectivity")?.unwrap_or(1.0),
                mix_ratio: r_list_real(params, "mix_ratio")?.unwrap_or(1.0),
            }),
            "knn" => Ok(Self::Knn {
                binary: params
                    .get("binary")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
            }),
            other => Err(Error::Other(format!(
                "Unknown knn_sym '{other}'; expected 'snn', 'umap' or 'knn'"
            ))),
        }
    }
}

impl FromRList for KnnSymmetrisation {
    /// Parse the [KnnSymmetrisation] from an R list; the keys are those of
    /// its [FromRMap] implementation.
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list to parse
    ///
    /// ### Returns
    ///
    /// The [KnnSymmetrisation]
    fn from_r_list(r_list: List) -> Result<Self> {
        Self::from_r_map(&r_list_to_map(r_list)?)
    }
}
