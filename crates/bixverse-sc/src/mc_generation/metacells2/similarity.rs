//! Cell-cell similarity computation.
//!
//! Operates on `pile.selected_dense` (cells × features). As upstream
//! (`direct.py`), adds `value_regularisation` to every value, optionally takes
//! `log2`, and correlates cells; `AbsPearson` (the default) then takes the
//! absolute value. The preprocessed matrix is written straight into the
//! features × cells layout `column_pairwise_cor` expects.

use faer::Mat;
use rayon::prelude::*;

use super::params::{SimilarityMethod, SimilarityParams};
use super::pile::Pile;

use crate::prelude::*;
use bixverse_core::core::base::cors_similarity::column_pairwise_cor;

/////////////
// Helpers //
/////////////

/// Regularise, optionally log, and transpose into features × cells.
///
/// ### Params
///
/// * `dense` - Input matrix of shape `(n_cells, n_features)`.
/// * `params` - Similarity parameters.
///
/// ### Returns
///
/// A `Mat<f32>` of shape `(n_features, n_cells)`.
fn preprocess(dense: &Mat<f32>, params: &SimilarityParams) -> Mat<f32> {
    let reg = params.value_regularisation;
    let mut out = Mat::<f32>::zeros(dense.ncols(), dense.nrows());
    out.par_col_iter_mut()
        .enumerate()
        .for_each(|(cell, mut col)| {
            for g in 0..col.nrows() {
                let v = dense[(cell, g)] + reg;
                col[g] = if params.log_data { v.log2() } else { v };
            }
        });
    out
}

//////////
// Main //
//////////

/// Compute the `n_cells × n_cells` similarity matrix from
/// `pile.selected_dense`.
///
/// ### Params
///
/// * `pile` - Pile with a populated `selected_dense`.
/// * `params` - Similarity parameters.
///
/// ### Returns
///
/// A symmetric `Mat<f32>` of shape `(n_cells, n_cells)`.
///
/// ### Errors
///
/// Returns `ScErrors::SelectFeaturesBeforeSimilariy` if
/// `pile.selected_dense` is `None`.
pub fn compute_similarity(pile: &Pile, params: &SimilarityParams) -> Result<Mat<f32>, ScErrors> {
    let dense = pile
        .selected_dense
        .as_ref()
        .ok_or(ScErrors::SelectFeaturesBeforeSimilariy)?;
    let preprocessed = preprocess(dense, params);
    let mut sim = column_pairwise_cor(&preprocessed.as_ref(), false);
    if params.method == SimilarityMethod::AbsPearson {
        sim.par_col_iter_mut().for_each(|col| {
            for v in col.iter_mut() {
                *v = v.abs();
            }
        });
    }
    Ok(sim)
}
