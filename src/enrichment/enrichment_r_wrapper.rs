//! R wrappers for the enrichment methods

use extendr_api::*;
use rustc_hash::FxHashMap;
use std::collections::HashMap;

use crate::enrichment::blitzgsea::{BlitzGseaNull, BlitzGseaParams};
use crate::enrichment::gsea::GseaParams;
use crate::enrichment::mitch::MitchPathways;
use crate::prelude::*;

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

////////////////
// blitzGSEA //
////////////////

impl BlitzGseaParams {
    /// Parse the [BlitzGseaParams] from a list
    ///
    /// A missing key falls back to the default, which is what the `None` arms of
    /// [`BlitzGseaParams::new`] resolve. Counts go through [`r_list_count`],
    /// which accepts either of R's storage modes, so `list(anchors = 40)` and
    /// `list(anchors = 40L)` both land.
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list to parse
    ///
    /// ### Returns
    ///
    /// The [BlitzGseaParams] populated by the R list.
    pub fn from_r_list(r_list: List) -> Result<Self> {
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
