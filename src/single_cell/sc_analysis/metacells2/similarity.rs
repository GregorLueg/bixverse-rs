//! Cell-cell similarity computation.
//!
//! Operates on `pile.selected_dense` (cells × features). For Pearson-based
//! methods, applies optional preprocessing (regularisation, `log2`) and
//! delegates to `column_pairwise_cor` after a layout transpose. For Spearman,
//! preprocessing is skipped — rank-based correlation is invariant to any
//! monotonic transform of the input, so `column_pairwise_cor` does its own
//! per-column ranking internally.
//!
//! ### Preprocessing semantics
//!
//! * `LogPearson` (default): `log2(x + 1 + reg)`. The hard-coded `+ 1`
//!   guarantees a finite, well-defined output regardless of `reg` or the
//!   presence of zeros. With the default `reg = 0` this is `log2(x + 1)`
//!   (log1p in log2 form), the canonical scRNA-seq choice for Pearson on
//!   counts.
//! * `Pearson`: `x + reg`. No log; behaves badly on raw counts and is
//!   provided mainly for parity testing.
//! * `Spearman`: no pre-processing needed.

use faer::Mat;

use super::params::{SimilarityMethod, SimilarityParams};
use super::pile::Pile;

use crate::core::base::cors_similarity::column_pairwise_cor;
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Apply method-specific preprocessing into a freshly allocated matrix.
///
/// Always allocates a full copy, including for `Spearman` where the
/// transform is a no-op, to keep the call site uniform. At typical pile
/// sizes (10K cells × 2K features) this is ~80 MB — dominated by the
/// `n_cells × n_cells` similarity matrix allocated downstream.
///
/// ### Params
///
/// * `dense`  - Input matrix of shape `(n_cells, n_features)`.
/// * `params` - Similarity parameters; reads `method` and `value_regularization`.
///
/// ### Returns
///
/// A new `Mat<f32>` of the same shape with the transform applied.
fn preprocess(dense: &Mat<f32>, params: &SimilarityParams) -> Mat<f32> {
    let reg = params.value_regularization;
    match params.method {
        SimilarityMethod::LogPearson => Mat::from_fn(dense.nrows(), dense.ncols(), |i, j| {
            (dense[(i, j)] + reg + 1.0).log2()
        }),
        SimilarityMethod::Pearson => {
            if reg == 0.0 {
                dense.clone()
            } else {
                Mat::from_fn(dense.nrows(), dense.ncols(), |i, j| dense[(i, j)] + reg)
            }
        }
        SimilarityMethod::Spearman => dense.clone(),
    }
}

//////////
// Main //
//////////

/// Compute the `n_cells × n_cells` similarity matrix from
/// `pile.selected_dense`.
///
/// Preprocesses the dense feature matrix according to `params.method`, then
/// delegates to `column_pairwise_cor` on the transposed `(features × cells)`
/// layout so that each cell is a column.
///
/// ### Params
///
/// * `pile`   - Pile with a populated `selected_dense`; see errors below.
/// * `params` - Similarity parameters; controls method and regularisation.
///
/// ### Returns
///
/// A `Mat<f32>` of shape `(n_cells, n_cells)` with pairwise correlation scores.
///
/// ### Errors
///
/// Returns `BixverseErrors::SelectFeaturesBeforeSimilariy` if
/// `pile.selected_dense` is `None`.
pub fn compute_similarity(
    pile: &Pile,
    params: &SimilarityParams,
) -> Result<Mat<f32>, BixverseErrors> {
    let dense = pile
        .selected_dense
        .as_ref()
        .ok_or(BixverseErrors::SelectFeaturesBeforeSimilariy)?;
    let preprocessed = preprocess(dense, params);
    let transposed = preprocessed.as_ref().transpose();
    let spearman = matches!(params.method, SimilarityMethod::Spearman);
    Ok(column_pairwise_cor(&transposed, spearman))
}
