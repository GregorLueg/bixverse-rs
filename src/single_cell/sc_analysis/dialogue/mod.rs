//! DIALOGUE: multicellular programmes from cross-cell-type association.
//!
//! Given several cell types profiled across shared samples or spatial niches,
//! finds programmes of co-regulated, cell-type-specific genes whose activity
//! covaries across those samples. Three stages:
//!
//! 1. Collapse each cell type's features to one row per sample and run sparse
//!    multi-CCA across the resulting blocks, giving each programme a weight
//!    vector per cell type and a provisional gene signature.
//! 2. For every ordered pair of cell types and every candidate gene, fit a
//!    mixed model asking whether a cell's own programme score tracks the
//!    partner cell type's expression of that gene in the same sample.
//! 3. Meta-analyse across partners and refit the scores onto the surviving
//!    genes by non-negative least squares.
//!
//! Everything the expression matrix is asked for is per-gene, so one
//! implementation bounded on [SingleCellReading] serves both streamed single
//! cells and in-memory metacells. See
//! [crate::single_cell::mc_analysis::dialogue_mc] for the metacell entry point.
//!
//! ### References
//!
//! Jerby-Arnon & Regev, Nature Biotechnology 40, 2022

pub mod params;
pub mod step1_pmd;
pub mod step2_hlm;
pub mod step3_refine;

use faer::MatRef;

use crate::prelude::*;

pub use params::{Averaging, DialogueParams, HlmParams, PmdParams, RefineParams, parse_averaging};
pub use step1_pmd::{ProgrammeSignature, Step1Result};
pub use step2_hlm::{GeneAssociation, Step2Result};
pub use step3_refine::{DialogueResult, GeneVerdict};

use step1_pmd::CellTypeView;

/// Runs the full DIALOGUE pipeline.
///
/// ### Params
///
/// * `reader` - Gene-major reader over the expression matrix
/// * `cell_type_indices` - Global cell indices per cell type
/// * `features` - Dense cell-level features per cell type, `n_cells_in_type x
///   k_i`, rows aligned to `cell_type_indices`. Principal components, factor
///   loadings, whatever the caller trusts; DIALOGUE does not compute them.
/// * `sample_ids` - Sample level code per global cell
/// * `cell_quality` - Quality covariate per global cell, upstream's `cellQ`.
///   Typically the z-scored log of the library size.
/// * `genes` - Genes to consider when building signatures
/// * `params` - See [DialogueParams]
/// * `verbose` - `0` silent, non-zero prints progress
///
/// ### Returns
///
/// The [DialogueResult], or the first error encountered. Common causes are too
/// few samples shared across cell types, too few features surviving the ANOVA
/// filter in one cell type, and a cell index outside the per-cell metadata.
#[allow(clippy::too_many_arguments)]
pub fn dialogue_run<S: SingleCellReading>(
    reader: &S,
    cell_type_indices: &[Vec<usize>],
    features: &[MatRef<f64>],
    sample_ids: &[usize],
    cell_quality: &[f64],
    genes: &[usize],
    params: &DialogueParams,
    verbose: usize,
) -> Result<DialogueResult, BixverseErrors> {
    let n_types = cell_type_indices.len();
    params.validate(n_types)?;
    if features.len() != n_types {
        return Err(BixverseErrors::InvalidArgument(format!(
            "got {n_types} cell types but {} feature matrices.",
            features.len()
        )));
    }
    for (i, (cells, feature)) in cell_type_indices.iter().zip(features.iter()).enumerate() {
        if feature.nrows() != cells.len() {
            return Err(BixverseErrors::ShapeMismatch {
                expected: (cells.len(), feature.ncols()),
                got: (feature.nrows(), feature.ncols()),
            });
        }
        if feature.ncols() < 2 {
            return Err(BixverseErrors::InvalidArgument(format!(
                "cell type {i} has {} feature columns; DIALOGUE needs at least 2.",
                feature.ncols()
            )));
        }
    }

    // Validated here rather than in `CellTypeView`, which only sees the
    // metadata: the dense position tables in stages one and three index by
    // global cell id and would panic past the end of the store. A panic across
    // the extendr boundary is worse than an error.
    let total_cells = reader.get_header().total_cells;
    for cells in cell_type_indices.iter() {
        if let Some(&bad) = cells.iter().find(|&&c| c >= total_cells) {
            return Err(BixverseErrors::InvalidArgument(format!(
                "cell index {bad} is outside the store's {total_cells} cells."
            )));
        }
    }

    let views: Vec<CellTypeView> = cell_type_indices
        .iter()
        .map(|cells| CellTypeView::new(cells, sample_ids, cell_quality))
        .collect::<Result<_, _>>()?;

    let step1 = step1_pmd::run_step1(reader, &views, features, genes, params, verbose)?;
    let step2 = step2_hlm::run_step2(reader, &views, &step1, params, verbose)?;
    step3_refine::run_step3(reader, &views, &step1, &step2, params, verbose)
}
