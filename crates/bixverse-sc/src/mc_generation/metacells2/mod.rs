//! Port of MetaCells2 (Ben-Kiki, et al., Genome Biol. 2022, metacells 0.9.5),
//! building on MetaCell (Baran, et al., Genome Biol. 2019). Two entry points:
//! [`compute_metacells_single_pile`] runs the direct pipeline on one pile, and
//! [`compute_dac_metacells`] divides and conquers for datasets beyond a pile.
//! Parity against the reference lives in `parity.rs`.

use crate::prelude::*;

pub mod candidates;
pub mod collect;
pub mod deviants;
pub mod direct;
pub mod dissolve;
pub mod divide_and_conquer;
pub mod downsample;
pub mod knn;
pub mod params;
pub mod partition;
pub mod pile;
pub mod rare;
pub mod seeds;
pub mod select;
pub mod similarity;

#[cfg(test)]
mod parity;
#[cfg(test)]
mod parity_fixtures;
#[cfg(test)]
mod parity_rare_fixtures;

// export the stuff
pub use candidates::{compute_candidate_metacells, make_incoming_view};
pub use deviants::find_deviant_cells;
pub use direct::{DirectMetacellsResult, compute_direct_metacells};
pub use dissolve::dissolve_metacells;
pub use divide_and_conquer::{
    DacMetacellsResult, compute_dac_metacells, compute_target_pile_size, random_piles,
};
pub use downsample::downsample_pile;
pub use knn::build_knn_graph;
pub use params::{
    DacParams, DeviantsParams, DissolveParams, MC2KnnParams, MetacellsParams, PartitionParams,
    RareParams, RareSimilarityMethod, SelectParams, SimilarityMethod, SimilarityParams,
};
pub use partition::{optimise_partitions, score_partitions};
pub use pile::Pile;
pub use rare::{RareGeneModules, find_rare_gene_modules};
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
/// The similarity and rank matrices are dense `n × n`, so beyond a pile's
/// worth of cells (upstream suggests around 10K) use [`compute_dac_metacells`].
///
/// ### Params
///
/// * `reader` - Reader for the cell-based store.
/// * `cell_indices` - Indices of the cells to include.
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
) -> Result<DirectMetacellsResult, ScErrors> {
    if verbose {
        println!("Loading {} cells into a single pile...", cell_indices.len());
    }

    let mut pile = Pile::from_reader(reader, cell_indices)?;

    compute_direct_metacells(&mut pile, params, seed, verbose)
}
