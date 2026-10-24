//! R wrappers for the enrichment methods

use extendr_api::*;
use rustc_hash::FxHashMap;
use std::collections::HashMap;

use crate::prelude::*;
use bixverse_rs::enrichment::blitzgsea::{BlitzGseaNull, BlitzGseaParams};
use bixverse_rs::enrichment::gsea::GseaParams;
use bixverse_rs::enrichment::lm_activity::{
    FeatureSetWeights, LmActivityParams, parse_lm_method, parse_lm_stat,
};
use bixverse_rs::enrichment::mitch::MitchPathways;

//////////
// GSVA //
//////////

/// Get gene set indices for GSVA
///
/// ### Params
///
/// * `gs_list` - R list that contains the different gene sets
///
/// ### Returns
///
/// A vector of vectors with the index positions as usizes
pub fn get_gsva_gs_indices(gs_list: List) -> Result<Vec<Vec<usize>>> {
    if gs_list.is_empty() {
        let gs_indices: Vec<Vec<usize>> = vec![vec![]];
        return Ok(gs_indices);
    }

    let mut gs_indices: Vec<Vec<usize>> = Vec::with_capacity(gs_list.len());

    for i in 0..gs_list.len() {
        let list_elem = gs_list.elt(i)?;
        let elem = list_elem
            .as_integer_vector()
            .unwrap()
            .iter()
            .map(|x| *x as usize)
            .collect();
        gs_indices.push(elem);
    }

    Ok(gs_indices)
}

//////////
// GSEA //
//////////

/// Prepare GSEA parameters from R list input
///
/// ### Params
///
/// * `r_list` - R list containing parameter values
///
/// ### Returns
///
/// `GseaParams` struct with parsed parameters (defaults: gsea_param=1.0,
/// min_size=5, max_size=500)
pub fn prepare_gsea_params<T: BixverseFloat>(r_list: List) -> Result<GseaParams<T>> {
    let gsea_params: HashMap<&str, Robj> = r_list_to_map(r_list)?;

    let gsea_param = gsea_params
        .get("gsea_param")
        .and_then(|v| v.as_real())
        .map(|v| T::from_f64(v).unwrap())
        .unwrap_or_else(|| T::one());

    let min_size = gsea_params
        .get("min_size")
        .and_then(|v| v.as_integer())
        .unwrap_or(5) as usize;

    let max_size = gsea_params
        .get("max_size")
        .and_then(|v| v.as_integer())
        .unwrap_or(500) as usize;

    Ok(GseaParams {
        gsea_param,
        max_size,
        min_size,
    })
}

///////////
// mitch //
///////////

/// Helper function to get the indices of the pathways
///
/// ### Params
///
/// * `row_names` - The row names of the matrix representing the represented
///   genes across all tested contrasts
/// * `pathway_list` - The named R list containing the pathway genes.
/// * `min_size` - The minimum overlap size
///
/// ### Returns
///
/// `MitchPathways = (Vec<String>, Vec<Vec<usize>>)` containing the pathway names
/// and their position
pub fn prepare_mitch_pathways(
    row_names: &[String],
    pathway_list: List,
    min_size: usize,
) -> Result<MitchPathways> {
    let gene_map: FxHashMap<&str, usize> = row_names
        .iter()
        .enumerate()
        .map(|(i, gene)| (gene.as_str(), i))
        .collect();

    let list_names: Vec<String> = pathway_list
        .names()
        .unwrap()
        .map(|s| s.to_string())
        .collect();

    let mut filtered_pathways = Vec::new();
    let mut filtered_names = Vec::new();

    for i in 0..pathway_list.len() {
        let element = pathway_list.elt(i)?;
        if let Some(internal_vals) = element.as_string_vector() {
            let mut indices = Vec::with_capacity(internal_vals.len());

            for gene in &internal_vals {
                if let Some(&idx) = gene_map.get(gene.as_str()) {
                    indices.push(idx);
                }
            }

            if indices.len() >= min_size {
                indices.sort_unstable();
                filtered_pathways.push(indices);
                filtered_names.push(list_names[i].clone());
            }
        }
    }

    Ok((filtered_names, filtered_pathways))
}

///////////////
// blitzGSEA //
///////////////

impl FromRList for BlitzGseaParams {
    /// Parse the [BlitzGseaParams] from a list
    ///
    /// A missing key falls back to the default, which is what the `None` arms
    /// of [`BlitzGseaParams::new`] resolve.
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list to parse
    ///
    /// ### Returns
    ///
    /// The [BlitzGseaParams] populated by the R list.
    fn from_r_list(r_list: List) -> Result<Self> {
        let params_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let flag = |key: &str| params_list.get(key).and_then(|v| v.as_bool());

        // R has no 64 bit integer type, so a seed arrives as a double
        let seed = params_list
            .get("seed")
            .and_then(|v| v.as_real())
            .map(|v| v as u64);

        Ok(Self::new(
            r_list_count(&params_list, "permutations")?,
            r_list_count(&params_list, "anchors")?,
            flag("symmetric"),
            flag("centre"),
            flag("ks_test"),
            seed,
        ))
    }
}

/// Serialise a calibrated null into an R list.
///
/// The null is under a kilobyte of plain numbers with no interior state, so it
/// crosses the boundary as data rather than as an external pointer. R can then
/// hold it in an environment, cache it against a hash of the signature, and
/// `saveRDS` it, none of which a pointer would survive.
///
/// ### Params
///
/// * `null` - The calibrated null from `calibrate_null`
///
/// ### Returns
///
/// A named R list carrying every field of the null.
pub fn blitzgsea_null_to_list(null: &BlitzGseaNull) -> List {
    list!(
        anchor_sizes = null.anchor_sizes.clone(),
        shape_pos = null.shape_pos.clone(),
        scale_pos = null.scale_pos.clone(),
        shape_neg = null.shape_neg.clone(),
        scale_neg = null.scale_neg.clone(),
        pos_ratio = null.pos_ratio.clone(),
        ks_pos = null.ks_pos,
        ks_neg = null.ks_neg,
        centred = null.centred
    )
}

/// Rebuild a calibrated null from the R list [`blitzgsea_null_to_list`] wrote.
///
/// Every parameter vector has to be the same length as the anchor grid, since
/// they are read in lockstep during interpolation. A list assembled by hand, or
/// one that has been subset R-side, would otherwise index out of bounds deep
/// inside the scoring loop.
///
/// ### Params
///
/// * `r_list` - The R list holding a serialised null
///
/// ### Returns
///
/// The reconstructed `BlitzGseaNull`, or an error naming the offending field.
pub fn blitzgsea_null_from_list(r_list: List) -> Result<BlitzGseaNull> {
    let fields: HashMap<&str, Robj> = r_list_to_map(r_list)?;

    let numeric = |key: &str| -> Result<Vec<f64>> {
        fields
            .get(key)
            .and_then(|v| v.as_real_vector())
            .ok_or_else(|| {
                extendr_api::Error::Other(format!(
                    "The blitzGSEA null model is missing a numeric '{key}'."
                ))
            })
    };

    let scalar = |key: &str| -> Result<f64> {
        fields.get(key).and_then(|v| v.as_real()).ok_or_else(|| {
            extendr_api::Error::Other(format!(
                "The blitzGSEA null model is missing a scalar '{key}'."
            ))
        })
    };

    let centred = fields
        .get("centred")
        .and_then(|v| v.as_bool())
        .ok_or_else(|| {
            extendr_api::Error::Other(
                "The blitzGSEA null model is missing a logical 'centred'.".to_string(),
            )
        })?;

    let anchor_sizes = numeric("anchor_sizes")?;
    if anchor_sizes.is_empty() {
        return Err(extendr_api::Error::Other(
            "The blitzGSEA null model has an empty anchor grid.".to_string(),
        ));
    }

    let checked = |key: &str| -> Result<Vec<f64>> {
        let values = numeric(key)?;
        if values.len() != anchor_sizes.len() {
            return Err(extendr_api::Error::Other(format!(
                "The blitzGSEA null model has {} values for '{key}' but {} anchor sizes.",
                values.len(),
                anchor_sizes.len()
            )));
        }
        Ok(values)
    };

    Ok(BlitzGseaNull {
        shape_pos: checked("shape_pos")?,
        scale_pos: checked("scale_pos")?,
        shape_neg: checked("shape_neg")?,
        scale_neg: checked("scale_neg")?,
        pos_ratio: checked("pos_ratio")?,
        ks_pos: scalar("ks_pos")?,
        ks_neg: scalar("ks_neg")?,
        centred,
        anchor_sizes,
    })
}

/////////////////
// LM activity //
/////////////////

impl FromRList for LmActivityParams {
    /// Generate LmActivityParams from an R list, falling back to defaults.
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list with `"method"` (`"ulm"`, `"mlm"`), `"stat"`
    ///   (`"t"`, `"coef"`) and `"p_values"`.
    ///
    /// ### Returns
    ///
    /// The populated [LmActivityParams]
    fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = Self::default();

        let method = match params.get("method").and_then(|v| v.as_str()) {
            Some(s) => parse_lm_method(s)
                .ok_or_else(|| Error::Other(format!("Invalid linear model method: {}", s)))?,
            None => defaults.method,
        };

        let stat = match params.get("stat").and_then(|v| v.as_str()) {
            Some(s) => parse_lm_stat(s)
                .ok_or_else(|| Error::Other(format!("Invalid linear model statistic: {}", s)))?,
            None => defaults.stat,
        };

        let p_values = params
            .get("p_values")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.p_values);

        Ok(Self {
            method,
            stat,
            p_values,
        })
    }
}

/// Transform an R list of weighted feature sets into `FeatureSetWeights`
///
/// ### Params
///
/// * `gs_list` - R list with one list per set holding `"idx"` (0-based
///   feature positions) and optionally `"weight"` (numeric, aligned with
///   `"idx"`). A set without weights gets weight 1 on every member.
/// * `n_features` - Size of the feature universe.
///
/// ### Returns
///
/// The feature sets
pub fn r_list_to_feature_set_weights(
    gs_list: List,
    n_features: usize,
) -> Result<FeatureSetWeights> {
    let mut sets: Vec<Vec<usize>> = Vec::with_capacity(gs_list.len());
    let mut weights: Vec<Vec<f64>> = Vec::with_capacity(gs_list.len());
    for i in 0..gs_list.len() {
        let set: HashMap<&str, Robj> = r_list_to_map(gs_list.elt(i)?.try_into()?)?;
        let idx: Vec<usize> = set
            .get("idx")
            .and_then(|v| v.as_integer_vector())
            .ok_or_else(|| Error::Other(format!("Feature set {} has no integer 'idx'.", i + 1)))?
            .iter()
            // NA_integer_ is i32::MIN, so this catches it too
            .map(|&x| usize::try_from(x))
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| {
                Error::Other(format!("Feature set {} has a negative or NA 'idx'.", i + 1))
            })?;
        // integer weights (e.g. a mode of regulation of 1L / -1L) are not a
        // real vector to extendr, so they need their own branch
        let w = match set.get("weight") {
            None => vec![1.0; idx.len()],
            Some(v) => match (v.as_real_vector(), v.as_integer_vector()) {
                (Some(w), _) => w,
                (None, Some(w)) if !w.contains(&i32::MIN) => w.iter().map(|&x| x as f64).collect(),
                _ => {
                    return Err(Error::Other(format!(
                        "Feature set {} has a non-numeric or NA 'weight'.",
                        i + 1
                    )));
                }
            },
        };
        sets.push(idx);
        weights.push(w);
    }
    FeatureSetWeights::new(&sets, Some(&weights), n_features).to_extendr()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use bixverse_rs::enrichment::lm_activity::lm_activity;
    use faer::Mat;

    #[test]
    fn test_feature_set_weights_accept_integer_weights() {
        test! {
            let as_int = List::from_values([list!(idx = r!([0, 3, 5]), weight = r!([1, -1, 1]))]);
            let as_real =
                List::from_values([list!(idx = r!([0, 3, 5]), weight = r!([1.0, -1.0, 1.0]))]);
            let mat = Mat::from_fn(10, 2, |i, j| ((i * 3 + j * 7) % 5) as f64);
            let a = lm_activity(mat.as_ref(), &r_list_to_feature_set_weights(as_int, 10)?, None)
                .unwrap();
            let b = lm_activity(mat.as_ref(), &r_list_to_feature_set_weights(as_real, 10)?, None)
                .unwrap();
            assert_eq!(a.es, b.es);
        }
    }

    #[test]
    fn test_feature_set_weights_reject_bad_input() {
        test! {
            let bad = [
                list!(idx = r!([0, -1, 5])),
                list!(idx = r!([0, 3, 5]), weight = r!([Some(1), NA_INTEGER, Some(1)])),
                list!(idx = r!([0, 3, 5]), weight = r!(["a", "b", "c"])),
            ];
            for set in bad {
                assert!(r_list_to_feature_set_weights(List::from_values([set]), 10).is_err());
            }
        }
    }
}
