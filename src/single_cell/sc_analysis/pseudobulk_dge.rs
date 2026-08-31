//! Pseudobulk differential expression through edgeR's quasi-likelihood chain.
//!
//! Sum the raw counts per sample, then treat the result as a bulk experiment.
//! That is the whole method, and it is the one that holds its nominal false
//! discovery rate when the cells within a sample are not independent, which
//! they never are.
//!
//! Both halves already exist:
//! [`pseudo_bulk_genes_dense`] returns genes by samples, which is the
//! orientation
//! [`run_edger_ql`](crate::methods::dge_bulk::run_edger_ql) wants, and the
//! numerics are `edge-rs`. This is the join.

use crate::methods::dge_bulk::{EdgeRDgeRes, EdgeRQlParams, run_edger_ql};
use crate::prelude::*;
use crate::single_cell::mc_generation::cell_aggregation_utils::{
    PseudoBulk, pseudo_bulk_genes_dense,
};

use edge_rs::glm::test::Tested;

/// Aggregates cells into samples and runs the edgeR quasi-likelihood test.
///
/// ### Params
///
/// * `reader` - Gene-major store the counts come from
/// * `gene_indices` - Genes to test. The result mask spans these, not the whole
///   store
/// * `sample_cells` - Cell indices per pseudobulk sample. Columns of the
///   aggregate follow this order, so the design rows must too
/// * `design` - Predictors, row-major `sample_cells.len() * n_coef`, including
///   an intercept
/// * `n_coef` - Number of design columns
/// * `tested` - Coefficients to drop from the null, or a contrast over them
/// * `params` - See [EdgeRQlParams]
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The [EdgeRDgeRes], with `genes_to_keep` indexed against `gene_indices`.
///
/// ### References
///
/// Squair et al., Nature Communications 12, 5692, 2021
#[allow(clippy::too_many_arguments)]
pub fn pseudobulk_dge<S: SingleCellReading>(
    reader: &S,
    gene_indices: &[usize],
    sample_cells: &[Vec<usize>],
    design: &[f64],
    n_coef: usize,
    tested: &Tested,
    params: &EdgeRQlParams,
    verbose: usize,
) -> Result<EdgeRDgeRes, BixverseErrors> {
    let n_samples = sample_cells.len();
    if design.len() != n_samples * n_coef {
        return Err(BixverseErrors::DgeShapeMismatch {
            name: "design",
            expected: n_samples * n_coef,
            got: design.len(),
        });
    }

    // Raw counts only. `PseudoBulk::Norm` averages the normalised layer over
    // every cell in the group, zeros included, which is no longer a count and
    // would put a negative binomial through something it cannot model.
    let aggregate =
        pseudo_bulk_genes_dense(reader, gene_indices, sample_cells, PseudoBulk::Raw, verbose)?;
    let counts = mat_to_flat_row_major(aggregate.as_ref());

    run_edger_ql(
        &counts,
        gene_indices.len(),
        n_samples,
        design,
        n_coef,
        tested,
        params,
    )
}
