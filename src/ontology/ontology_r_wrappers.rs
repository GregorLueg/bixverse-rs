use extendr_api::*;
use std::collections::BTreeMap;

use crate::ontology::go_elim::*;
use crate::prelude::*;

///////////////////////
// Onto similarities //
///////////////////////

/// Transform an R list that hopefully contains the IC into a HashMap of floats
///
/// ### Params
///
/// * `r_list` - The R list with the IC values
///
/// ### Returns
///
/// The BTreeMap with the information content values
pub fn ic_list_to_ic_hashmap(r_list: List) -> BTreeMap<String, f64> {
    let mut hashmap = BTreeMap::new();
    for (name, x) in r_list {
        let name = name.to_string();
        let ic_val = x.as_real().unwrap_or(0.0);
        hashmap.insert(name, ic_val);
    }

    hashmap
}

////////////////////////
// Gene ontology elim //
////////////////////////

/// Transform the S7 class into the needed data
///
/// ### Params
///
/// * `go_obj` - The S7 class storing the Gene Ontology data for the elimination methods
///
/// ### Returns
///
/// A tuple with the `GeneMap`, `AncestorMap` and `LevelMap`.
pub fn prepare_go_data(go_obj: Robj) -> (GeneMap, AncestorMap, LevelMap) {
    // TODO: Need to do better error handling here... Future me problem
    let go_to_genes = go_obj.get_attrib("go_to_genes").unwrap().as_list().unwrap();
    let ancestors = go_obj.get_attrib("ancestry").unwrap().as_list().unwrap();
    let levels = go_obj.get_attrib("levels").unwrap().as_list().unwrap();

    let go_to_genes = r_list_to_hashmap_set(go_to_genes).unwrap();
    let ancestors = r_list_to_hashmap(ancestors).unwrap();
    let levels = r_list_to_hashmap(levels).unwrap();

    (go_to_genes, ancestors, levels)
}
