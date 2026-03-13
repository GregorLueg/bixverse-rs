//! R wrappers for the enrichment methods

use extendr_api::*;
use rustc_hash::FxHashMap;

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
pub fn prepare_gsea_params<T: BixverseFloat>(r_list: List) -> GseaParams<T> {
    let gsea_params = r_list.into_hashmap();

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

    GseaParams {
        gsea_param,
        max_size,
        min_size,
    }
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
