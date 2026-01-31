use extendr_api::*;

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
