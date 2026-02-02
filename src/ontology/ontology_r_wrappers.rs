use extendr_api::List;
use std::collections::BTreeMap;

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
