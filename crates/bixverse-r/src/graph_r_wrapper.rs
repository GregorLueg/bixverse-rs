//! R wrappers over structures within the `graph` module

use extendr_api::*;
use std::collections::HashMap;

use crate::prelude::*;
use bixverse_rs::graph::knn_graph::{
    KnnSymmetrisation, SNN_PRUNE_DEFAULT, SnnSimilarity, parse_snn_similarity,
};
use bixverse_rs::graph::leiden::{LeidenParams, LeidenParamsSpec};

//////////////////
// LeidenParams //
//////////////////

impl FromRList for LeidenParams {
    /// Parse the [LeidenParams] from an R list.
    ///
    /// Missing keys fall back to [LeidenParams::for_algorithm] of `refine`.
    /// Recognised keys: `resolution`, `beta`, `n_iterations` (negative: until
    /// stable), `min_gain`, `refine`, `seed`, `local_moving` (`"queue"` or
    /// `"coloured"`) and `node_order` (`"bfs"` or `"shuffle"`).
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
        let d = LeidenParamsSpec::default();
        let string = |key: &str| params.get(key).and_then(|v| v.as_str()).map(str::to_string);
        let spec = LeidenParamsSpec {
            resolution: r_list_real(&params, "resolution")?.unwrap_or(d.resolution),
            beta: r_list_real(&params, "beta")?.unwrap_or(d.beta),
            n_iterations: r_list_real(&params, "n_iterations")?,
            min_gain: r_list_real(&params, "min_gain")?.unwrap_or(d.min_gain),
            refine: params
                .get("refine")
                .and_then(|v| v.as_bool())
                .unwrap_or(d.refine),
            seed: r_list_count_allow_zero(&params, "seed")?.map_or(d.seed, |v| v as u64),
            local_moving: string("local_moving"),
            node_order: string("node_order"),
        };
        Self::try_from(spec).map_err(Error::Other)
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
                    _ => (SNN_PRUNE_DEFAULT, true),
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
