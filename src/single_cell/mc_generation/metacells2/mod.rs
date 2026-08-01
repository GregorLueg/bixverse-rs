//! Port of the MetaCells2 workflow over into `biverse-rs`. The original
//! implementations were in R/C++ for the original implementation by Baran,
//! et al., Genome Biol. 2019; this was expanded in the MetaCells2 framework
//! into Python and C++, please refer to Ben‑Kiki, et al., Genome Biol. 2022.
//! Due to complexity of the algorithm, this lives in its own module.

use crate::prelude::*;

pub mod candidates;
pub mod deviants;
pub mod direct;
pub mod dissolve;
pub mod downsample;
pub mod knn;
pub mod params;
pub mod partition;
pub mod pile;
pub mod seeds;
pub mod select;
pub mod similarity;

// export the stuff
pub use candidates::{compute_candidate_metacells, make_incoming_view};
pub use deviants::find_deviant_cells;
pub use direct::{DirectMetacellsResult, compute_direct_metacells};
pub use dissolve::dissolve_metacells;
pub use downsample::downsample_pile;
pub use knn::build_knn_graph;
pub use params::{
    DeviantsParams, DissolveParams, MC2KnnParams, MetacellsParams, PartitionParams, SelectParams,
    SimilarityMethod, SimilarityParams,
};
pub use partition::{optimise_partitions, score_partitions};
pub use pile::Pile;
pub use seeds::{choose_seeds, seeds_count_for};
pub use select::select_features;
pub use similarity::compute_similarity;

/////////////////
// Single pile //
/////////////////

/// Run the direct MC2 pipeline on the entire dataset as a single pile.
///
/// Convenience wrapper for the case where the dataset fits in memory and
/// no divide-and-conquer is needed. Loads all cells via the reader,
/// constructs a single [`Pile`], runs [`compute_direct_metacells`], and
/// returns the result.
///
/// For datasets larger than RAM or where the cell count is much greater
/// than the practical optimiser/similarity ceiling (~10K cells), use the
/// divide-and-conquer entry point instead (not yet implemented).
///
/// ### Params
///
/// * `reader` - Reader for the cell-based store.
/// * `cells_to_include` - Indices of the cells to include.
/// * `params` - MC2 parameters.
/// * `seed` - Master RNG seed.
/// * `verbose` - If true, prints per-stage timing to stdout.
///
/// ### Returns
///
/// A [`DirectMetacellsResult`] with per-cell assignments and flags. The
/// indices in the returned vectors correspond to the reader's cell
/// indices in their natural order.
pub fn compute_metacells_single_pile<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    params: &MetacellsParams,
    seed: usize,
    verbose: bool,
) -> Result<DirectMetacellsResult, BixverseErrors> {
    if verbose {
        println!("Loading {} cells into a single pile...", cell_indices.len());
    }

    let mut pile = Pile::from_reader(reader, cell_indices)?;

    compute_direct_metacells(&mut pile, params, seed, verbose)
}
