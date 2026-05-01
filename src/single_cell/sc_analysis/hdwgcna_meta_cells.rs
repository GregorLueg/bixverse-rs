//! Meta-cell aggregation based on the bootstrapped approach from Morabito,
//! et al., Cell Rep. Methods, 2023

use rand::prelude::IndexedRandom;
use rand::{Rng, SeedableRng};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::prelude::*;

////////////////////////
// hdWGCNA meta cells //
////////////////////////

/// Structure for the MetaCell parameters
#[derive(Clone, Debug)]
pub struct MetaCellParams {
    /// Maximum number of shared cells for the meta cell aggregation
    pub max_shared: usize,
    /// Number of target meta cells.
    pub target_no_metacells: usize,
    /// Maximum iterations for the algorithm
    pub max_iter: usize,
    /// Parameters for the various approximate nearest neighbour searches
    /// in ann-search-rs
    pub knn_params: KnnParams,
}

/// Select meta cells
///
/// ### Params
///
/// * `nn_map` - Nearest neighbours with self
/// * `max_shared` - Maximum number of shared neighbours to allow
/// * `target_no` - Target number of meta cells
/// * `max_iter` - Maximum iterations for the algorithm
/// * `seed` - seed for reproducibility purposes
///
/// ### Returns
///
/// Borrowed slice of the selected meta cell indices
pub fn identify_meta_cells(
    nn_map: &[Vec<usize>],
    max_shared: usize,
    target_no: usize,
    max_iter: usize,
    seed: usize,
    verbose: bool,
) -> Vec<usize> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
    let k = nn_map[0].len();
    let k2 = k * 2;
    let mut good_choices: Vec<usize> = (0..nn_map.len()).collect();
    let mut chosen: Vec<usize> = Vec::new();

    if let Some(&first) = good_choices.choose(&mut rng) {
        chosen.push(first);
        good_choices.retain(|&x| x != first);
    }

    let mut it = 0;
    let mut set_cache: FxHashMap<usize, FxHashSet<usize>> = FxHashMap::default();

    while !good_choices.is_empty() && chosen.len() < target_no && it < max_iter {
        it += 1;
        let choice_idx = rng.random_range(0..good_choices.len());
        let candidate = good_choices.swap_remove(choice_idx);

        set_cache
            .entry(candidate)
            .or_insert_with(|| nn_map[candidate].iter().copied().collect());

        let mut max_overlap = 0;
        for &existing in &chosen {
            set_cache
                .entry(existing)
                .or_insert_with(|| nn_map[existing].iter().copied().collect());
            let cs = &set_cache[&candidate];
            let es = &set_cache[&existing];
            let shared = k2 - cs.union(es).count();
            max_overlap = max_overlap.max(shared);
        }

        if verbose && it % 10000 == 0 {
            println!(
                "Meta cell neighbour search - iter {} of {} max iters",
                it, max_iter
            );
        }

        if max_overlap <= max_shared {
            chosen.push(candidate);
        }
    }

    chosen
}

/// Assign cells to bootstrapped metacells with centre priority.
///
/// Centres always claim themselves first, so no metacell is empty. Remaining
/// cells go to the first centre whose kNN window contains them.
///
/// ### Params
///
/// * `centres` - The center cell.
/// * `nn_map` - The nearest neighbour map
///
/// ### Returns
///
/// The meta cell assignments
pub fn assign_bootstrapped_meta_cells(
    centres: &[usize],
    nn_map: &[Vec<usize>],
) -> Vec<Option<usize>> {
    let mut assignments = vec![None; nn_map.len()];

    for (mc_id, &c) in centres.iter().enumerate() {
        assignments[c] = Some(mc_id);
    }

    for (mc_id, &c) in centres.iter().enumerate() {
        for &nb in &nn_map[c] {
            if assignments[nb].is_none() {
                assignments[nb] = Some(mc_id);
            }
        }
    }

    assignments
}
