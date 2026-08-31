//! NEBULA for meta cells, see He et al., Commun Biol, 2021.
//!
//! The algorithm is the single-cell one verbatim:
//! [`run_nebula`](crate::single_cell::sc_analysis::nebula::run_nebula) is
//! generic over [`SingleCellReading`] and needs gene chunks plus per-cell
//! library sizes, both of which [`InMemorySparseReader`] serves out of a
//! [`CompressedSparseData2`]. This entry point is the thin shim that builds
//! that reader and hands it over.
//!
//! What changes is the interpretation, not the arithmetic. NEBULA splits the
//! variance into a subject-level random effect and a cell-level
//! overdispersion. Run it on metacells and the cell-level term is the spread
//! between aggregates within a subject, not between cells, so it is smaller
//! and it absorbs whatever the aggregation smoothed away. The subject-level
//! term keeps its meaning. Treat the two as a variance decomposition over
//! metacells and do not compare them against a single-cell run.

use crate::prelude::*;
use crate::single_cell::mc_analysis::as_csc;
use crate::single_cell::sc_analysis::nebula::{NebulaScParams, NebulaScRes, run_nebula};
use crate::single_cell::sc_data::in_memory_io::InMemorySparseReader;

/// Fits NEBULA to every requested gene over a metacell matrix.
///
/// ### Params
///
/// * `matrix` - The metacell counts, shape (metacells, genes). Raw counts in
///   `data`, normalised counts in `data_2`. Either orientation is accepted.
/// * `metacells_to_keep` - Indices of the metacells to analyse, in any order
/// * `genes_to_use` - Indices of the genes to fit
/// * `subject_ids` - Subject label per metacell, one entry per row of `matrix`
/// * `design` - Predictors, row-major `metacells_to_keep.len() * n_coef`, rows
///   aligned to `metacells_to_keep` and including an intercept
/// * `n_coef` - Number of design columns
/// * `offset` - Strictly positive scaling factor per selected metacell, or
///   `None` to use the aggregated library sizes
/// * `params` - See [NebulaScParams]
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The [NebulaScRes].
///
/// ### References
///
/// He et al., Communications Biology 4, 629, 2021
#[allow(clippy::too_many_arguments)]
pub fn nebula_metacells(
    matrix: &CompressedSparseData2<u32, f32>,
    metacells_to_keep: &[usize],
    genes_to_use: &[usize],
    subject_ids: &[usize],
    design: &[f64],
    n_coef: usize,
    offset: Option<&[f64]>,
    params: &NebulaScParams,
    verbose: usize,
) -> Result<NebulaScRes, BixverseErrors> {
    let csc = as_csc(matrix);
    let reader = InMemorySparseReader::new(csc.as_ref(), None)?;

    run_nebula(
        &reader,
        &reader,
        metacells_to_keep,
        genes_to_use,
        subject_ids,
        design,
        n_coef,
        offset,
        params,
        verbose,
    )
}
