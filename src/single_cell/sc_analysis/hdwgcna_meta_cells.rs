use rand::prelude::IndexedRandom;
use rand::{Rng, SeedableRng};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::prelude::*;

////////////////////////
// hdWGCNA meta cells //
////////////////////////

/// Structure for the MetaCell parameters
///
/// ### Fields
///
/// ** Meta cell params**
///
/// * `max_shared` - Maximum number of shared cells for the meta cell
///   aggregation
/// * `target_no_metacells` - Number of target meta cells.
/// * `max_iter` - Maximum iterations for the algorithm.
/// * `knn_params` - The knnParams via the `KnnParams` structure.
#[derive(Clone, Debug)]
pub struct MetaCellParams {
    // meta cell params
    pub max_shared: usize,
    pub target_no_metacells: usize,
    pub max_iter: usize,
    // general knn params
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
) -> Vec<&[usize]> {
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

    // cache the HashSets to avoid regeneration during loops
    let mut set_cache: FxHashMap<usize, FxHashSet<usize>> = FxHashMap::default();

    // bootstrap meta cells
    while !good_choices.is_empty() && chosen.len() < target_no && it < max_iter {
        it += 1;

        // sample remaining cells
        let choice_idx = rng.random_range(0..good_choices.len());
        let candidate = good_choices[choice_idx];
        good_choices.remove(choice_idx);

        // check overlap with existing meta cells
        set_cache
            .entry(candidate)
            .or_insert_with(|| nn_map[candidate].iter().copied().collect());

        let mut max_overlap = 0;
        for &existing in &chosen {
            set_cache
                .entry(existing)
                .or_insert_with(|| nn_map[existing].iter().copied().collect());

            let candidate_set = &set_cache[&candidate];
            let existing_set = &set_cache[&existing];

            let shared = k2 - candidate_set.union(existing_set).count();
            max_overlap = max_overlap.max(shared);
        }

        if verbose && it % 10000 == 0 {
            println!(
                "Meta cell neighbour search - iter {} out of {} max iters",
                it, max_iter
            );
        }

        if max_overlap <= max_shared {
            chosen.push(candidate);
        }
    }

    chosen
        .iter()
        .map(|&center| nn_map[center].as_slice())
        .collect()
}
