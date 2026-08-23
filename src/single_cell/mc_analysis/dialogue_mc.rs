//! DIALOGUE over in-memory metacells.
//!
//! A shim, not a second implementation. Everything DIALOGUE asks of the
//! expression matrix is per-gene, so the streaming core in
//! [crate::single_cell::sc_analysis::dialogue] runs unchanged once the metacell
//! matrix is wrapped in an [InMemorySparseReader].
//!
//! The only thing to keep in mind is what a "cell" means here. Metacells are
//! already aggregates, so the sample a metacell belongs to has to be
//! unambiguous: build them within samples, not across them. The random
//! intercept in stage two is over samples, and a metacell straddling two of
//! them has no well-defined level.

use faer::MatRef;

use crate::prelude::*;
use crate::single_cell::mc_analysis::as_csc;
use crate::single_cell::sc_analysis::dialogue::{DialogueParams, DialogueResult, dialogue_run};
use crate::single_cell::sc_data::in_memory_io::InMemorySparseReader;

/// Runs DIALOGUE on metacells held in memory.
///
/// ### Params
///
/// * `matrix` - Metacell counts, shape (metacells, genes). Raw counts in
///   `data`, normalised counts in `data_2`. Either orientation is accepted.
/// * `cell_type_indices` - Metacell indices per cell type
/// * `features` - Dense metacell-level features per cell type,
///   `n_metacells_in_type x k_i`, rows aligned to `cell_type_indices`
/// * `sample_ids` - Sample level code per metacell
/// * `cell_quality` - Quality covariate per metacell
/// * `genes` - Genes to consider when building signatures
/// * `params` - See [DialogueParams]
/// * `verbose` - `0` silent, non-zero prints progress
///
/// ### Returns
///
/// The [DialogueResult], or the first error encountered.
#[allow(clippy::too_many_arguments)]
pub fn dialogue_metacells(
    matrix: &CompressedSparseData2<u32, f32>,
    cell_type_indices: &[Vec<usize>],
    features: &[MatRef<f64>],
    sample_ids: &[usize],
    cell_quality: &[f64],
    genes: &[usize],
    params: &DialogueParams,
    verbose: usize,
) -> Result<DialogueResult, BixverseErrors> {
    let csc = as_csc(matrix);
    let reader = InMemorySparseReader::new(csc.as_ref(), None)?;
    dialogue_run(
        &reader,
        cell_type_indices,
        features,
        sample_ids,
        cell_quality,
        genes,
        params,
        verbose,
    )
}
