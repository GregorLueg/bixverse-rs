//! Candidate metacell orchestration.
//!
//! Faithful port of upstream MC2's `compute_candidate_metacells`. Wraps the
//! SA partition optimiser (`partition.rs`) in an outer loop that:
//!
//! 1. Optimises an initial seed partition.
//! 2. Splits oversized partitions (random multi-way) and cuts loosely-coupled
//!    partitions (Stoer-Wagner min-cut), reseeds the affected cells, and
//!    re-optimises. Repeats until no further splits/cuts.
//! 3. Detects small partitions, dissolves them to outliers, reseeds the
//!    leftover cells, and re-optimises. Repeats until either no more
//!    small partitions exist or the score stops improving.
//!
//! ### Outputs
//!
//! Returns a fully populated `partition_of_cells: Vec<i32>` where every
//! entry is `>= 0` and IDs are dense `0..n_partitions`. On exit the inner
//! loop has converged and downstream stages (deviants, dissolve, projection)
//! can run on the result.

use std::collections::BTreeSet;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustc_hash::FxHashSet;

use crate::core::math::sparse::transpose_sparse;
use crate::prelude::*;

use super::params::MetacellsParams;
use super::partition::optimise_partitions;
use super::seeds::{choose_seeds, seeds_count_for};

////////////
// Consts //
////////////

/// Increase the upper-size threshold by 1% per outer phase. Mirrors upstream's
/// `increase_phase` constant.
const INCREASE_PHASE: f64 = 1.01;

///////////
// Enums //
///////////

/// Outcome of a min-cut attempt on a single community.
#[derive(PartialEq, Eq, Debug)]
enum CutAction {
    /// The community was left unchanged (cut too strong, or already atomic).
    Unchanged,
    /// The community was split into two new communities.
    Split,
    /// The small side was demoted to outliers; the large side remains.
    Cut,
}

/////////////
// Helpers //
/////////////

/// Inner phase loop: alternate SA optimisation and cut/split passes until
/// neither produces any changes.
///
/// On each iteration, optimises the current partition, nudges
/// `max_metacell_size` upward by `INCREASE_PHASE`, then runs
/// `cut_split_communities`. If no splits or cuts occurred, the partition has
/// converged and the loop exits. Otherwise the demoted cells are reseeded and
/// the loop repeats.
///
/// ### Params
///
/// * `outgoing`, `incoming` - CSR graph and incoming-neighbour transpose.
/// * `partition_of_cells` - Current assignment; modified in place.
/// * `hot_communities` - Communities eligible for SA moves; updated in place.
/// * `cell_umis` - Per-cell UMI counts.
/// * `min_metacell_umis`, `max_metacell_umis` - UMI budget bounds.
/// * `target_umis` - Ideal per-metacell UMI count.
/// * `min_metacell_size` - Minimum acceptable cell count per metacell.
/// * `target_size` - Ideal cell count per metacell.
/// * `max_metacell_size` - Current upper cell-count bound; grown each
///   iteration.
/// * `params` - Top-level MC2 parameters.
/// * `cold_temperature` - Starting SA temperature for frozen communities.
/// * `atomic_candidates` - Memoisation set for min-cut cell sets already tried.
/// * `random_seed` - Master RNG seed.
///
/// ### Returns
///
/// `(cold_temperature, score)` after convergence.
#[allow(clippy::too_many_arguments)]
fn reduce_communities(
    outgoing: &CompressedSparseData2<f32, f32>,
    incoming: &CompressedSparseData2<f32, f32>,
    partition_of_cells: &mut [i32],
    hot_communities: &mut Vec<usize>,
    cell_umis: &[f32],
    min_metacell_umis: f64,
    max_metacell_umis: f64,
    target_umis: f64,
    min_metacell_size: usize,
    target_size: usize,
    max_metacell_size: &mut f64,
    params: &MetacellsParams,
    mut cold_temperature: f64,
    atomic_candidates: &mut BTreeSet<Vec<usize>>,
    random_seed: u64,
) -> (f64, f64) {
    loop {
        let n_partitions = max_partition_id(partition_of_cells) + 1;
        let mut hot_mask = vec![false; n_partitions];
        for &p in hot_communities.iter() {
            if p < hot_mask.len() {
                hot_mask[p] = true;
            }
        }

        let score = optimise_partitions(
            outgoing,
            incoming,
            cell_umis,
            partition_of_cells,
            min_metacell_umis,
            target_umis,
            max_metacell_umis,
            min_metacell_size as f64,
            target_size as f64,
            *max_metacell_size,
            &params.partition,
            cold_temperature,
            hot_mask,
            random_seed,
        );

        cold_temperature *= 1.0 - params.partition.cooldown_phase;
        *max_metacell_size *= INCREASE_PHASE;

        let (split_count, cut_count, new_hot) = cut_split_communities(
            outgoing,
            partition_of_cells,
            cell_umis,
            min_metacell_umis,
            max_metacell_umis,
            target_umis,
            min_metacell_size,
            target_size,
            *max_metacell_size,
            params.partition.max_split_min_cut_strength,
            params.partition.min_cut_seed_cells,
            params.must_complete_cover,
            atomic_candidates,
            random_seed,
        );

        *hot_communities = new_hot;

        if split_count + cut_count == 0 {
            return (cold_temperature, score);
        }

        // Reseed only the cells that were demoted to -1 by splits/cuts.
        let mut max_seeds_count = max_partition_id(partition_of_cells) + 1;
        if cut_count > 0 {
            // A cut creates one extra seed slot for the dropped cells to
            // glue onto something other than the originating community.
            hot_communities.push(max_seeds_count);
            max_seeds_count += 1;
        }
        choose_seeds(
            outgoing,
            incoming,
            partition_of_cells,
            max_seeds_count,
            params.partition.min_seed_size_quantile,
            params.partition.max_seed_size_quantile,
            random_seed,
        );
    }
}

/// Run one round of random splits (Phase A) and min-cut splitting (Phase B).
///
/// Phase A splits each community whose size or UMI total is above the current
/// maximum into random k-way parts, demoting all but one seed cell per part to
/// `-1`. If any Phase A splits occurred, Phase B is skipped.
///
/// Phase B runs Stoer-Wagner min-cut on each community. A low-strength cut is
/// accepted as a split if the smaller partition meets `min_cut_seed_cells`;
/// otherwise the small side is demoted to outliers (cut), unless
/// `must_complete_cover` prevents it.
///
/// ### Params
///
/// * `outgoing` - CSR outgoing-neighbour graph.
/// * `partition_of_cells` - Current assignment; modified in place.
/// * `cell_umis` - Per-cell UMI counts.
/// * `min_metacell_umis`, `max_metacell_umis` - UMI budget bounds.
/// * `target_umis` - Ideal UMI count per metacell.
/// * `min_metacell_size` - Minimum acceptable cell count per metacell.
/// * `target_size` - Ideal cell count per metacell.
/// * `max_metacell_size` - Current upper cell-count bound.
/// * `max_split_min_cut_strength` - Cut-strength threshold; cuts above this are
///   rejected.
/// * `min_cut_seed_cells` - Minimum cells for the small side to be a split
///   rather than a cut.
/// * `must_complete_cover` - If `true`, never demote cells to outliers.
/// * `atomic_candidates` - Memoisation set; cell sets already tried are
///   skipped.
/// * `random_seed` - RNG seed.
///
/// ### Returns
///
/// `(split_count, cut_count, new_hot_communities)`.
#[allow(clippy::too_many_arguments)]
fn cut_split_communities(
    outgoing: &CompressedSparseData2<f32, f32>,
    partition_of_cells: &mut [i32],
    cell_umis: &[f32],
    min_metacell_umis: f64,
    max_metacell_umis: f64,
    target_umis: f64,
    min_metacell_size: usize,
    target_size: usize,
    max_metacell_size: f64,
    max_split_min_cut_strength: f64,
    min_cut_seed_cells: usize,
    must_complete_cover: bool,
    atomic_candidates: &mut BTreeSet<Vec<usize>>,
    random_seed: u64,
) -> (usize, usize, Vec<usize>) {
    let n_partitions = max_partition_id(partition_of_cells) + 1;
    let mut split_count = 0usize;
    let mut cut_count = 0usize;
    let mut next_new = n_partitions;
    let mut hot_communities: Vec<usize> = Vec::new();
    let mut rng = StdRng::seed_from_u64(random_seed.wrapping_add(0x5EED5_5EED5_5EED5));

    // Phase A: random splits.
    for community_index in 0..n_partitions {
        let cells: Vec<usize> = (0..partition_of_cells.len())
            .filter(|&i| partition_of_cells[i] == community_index as i32)
            .collect();
        if cells.is_empty() {
            continue;
        }
        let community_size = cells.len();
        let community_umis: f64 = cells.iter().map(|&i| cell_umis[i] as f64).sum();

        let too_large_by_umis =
            community_umis > max_metacell_umis && community_size >= 3 * min_metacell_size;
        let too_large_by_size =
            community_size as f64 > max_metacell_size && community_umis >= 3.0 * min_metacell_umis;

        if !(too_large_by_umis || too_large_by_size) {
            continue;
        }

        let split_parts_count = ((community_umis / target_umis).round() as usize)
            .min((community_size as f64 / target_size as f64).round() as usize)
            .max(2);

        // Repeat assignment until every part is non-empty (statistically
        // almost always one shot, but cheap to verify).
        let mut assignment;
        loop {
            assignment = (0..cells.len())
                .map(|_| rng.random_range(0..split_parts_count))
                .collect::<Vec<usize>>();
            if (0..split_parts_count).all(|s| assignment.contains(&s)) {
                break;
            }
        }

        // Demote everyone, then re-seed one cell per part.
        for &c in &cells {
            partition_of_cells[c] = -1;
        }
        for split_index in 0..split_parts_count {
            let in_part: Vec<usize> = cells
                .iter()
                .copied()
                .zip(assignment.iter().copied())
                .filter_map(|(c, s)| if s == split_index { Some(c) } else { None })
                .collect();
            let seed_cell = in_part[rng.random_range(0..in_part.len())];
            let target_id = if split_index == 0 {
                community_index
            } else {
                let id = next_new;
                next_new += 1;
                split_count += 1;
                id
            };
            partition_of_cells[seed_cell] = target_id as i32;
            hot_communities.push(target_id);
        }
    }

    if split_count > 0 {
        return (split_count, cut_count, hot_communities);
    }

    // Phase B: min-cut.
    for community_index in 0..n_partitions {
        let cells: Vec<usize> = (0..partition_of_cells.len())
            .filter(|&i| partition_of_cells[i] == community_index as i32)
            .collect();
        if cells.len() < 2 {
            continue;
        }

        // memoise: skip if we already tried this exact cell set.
        let mut key = cells.clone();
        key.sort_unstable();
        if atomic_candidates.contains(&key) {
            continue;
        }

        let action = min_cut_community(
            outgoing,
            partition_of_cells,
            &cells,
            max_split_min_cut_strength,
            min_cut_seed_cells,
            must_complete_cover,
            next_new as i32,
        );

        match action {
            CutAction::Unchanged => {
                atomic_candidates.insert(key);
            }
            CutAction::Split => {
                hot_communities.push(community_index);
                hot_communities.push(next_new);
                next_new += 1;
                split_count += 1;
            }
            CutAction::Cut => {
                hot_communities.push(community_index);
                cut_count += 1;
            }
        }
    }

    (split_count, cut_count, hot_communities)
}

/// Run Stoer-Wagner min-cut on the induced subgraph of `cells` and apply
/// the result to `partition_of_cells`.
///
/// If the cut strength exceeds `max_split_min_cut_strength`, the community is
/// left unchanged. Otherwise the smaller partition is either promoted to a new
/// community (if `>= min_cut_seed_cells` cells) or demoted to outliers.
///
/// ### Params
///
/// * `outgoing` - CSR outgoing-neighbour graph.
/// * `partition_of_cells` - Current assignment; modified in place.
/// * `cells` - Global cell indices belonging to this community.
/// * `max_split_min_cut_strength` - Strength threshold above which the cut is
///   rejected.
/// * `min_cut_seed_cells` - Minimum cells in the small partition to accept a
///   split.
/// * `must_complete_cover` - If `true`, never demote cells to outliers.
/// * `new_community_index` - Partition ID to assign to the larger half on a
///   split.
///
/// ### Returns
///
/// A `CutAction` describing the outcome.
#[allow(clippy::too_many_arguments)]
fn min_cut_community(
    outgoing: &CompressedSparseData2<f32, f32>,
    partition_of_cells: &mut [i32],
    cells: &[usize],
    max_split_min_cut_strength: f64,
    min_cut_seed_cells: usize,
    must_complete_cover: bool,
    new_community_index: i32,
) -> CutAction {
    if cells.len() < 2 {
        return CutAction::Unchanged;
    }

    let (cut_strength, partition_a, partition_b) = match stoer_wagner(outgoing, cells) {
        Some(r) => r,
        None => return CutAction::Unchanged,
    };

    if cut_strength > max_split_min_cut_strength {
        return CutAction::Unchanged;
    }

    let (small, large) = if partition_a.len() <= partition_b.len() {
        (partition_a, partition_b)
    } else {
        (partition_b, partition_a)
    };

    if small.len() >= min_cut_seed_cells {
        for &c in &large {
            partition_of_cells[c] = new_community_index;
        }
        return CutAction::Split;
    }

    if must_complete_cover && cut_strength > 0.0 {
        return CutAction::Unchanged;
    }

    // demote the small side to outliers.
    for &c in &small {
        partition_of_cells[c] = -1;
    }
    CutAction::Cut
}

/// Global min-cut via Stoer-Wagner on the induced subgraph of `cells`.
///
/// Builds a symmetric local weight matrix, runs maximum-adjacency ordering
/// `n - 1` times, and tracks the minimum cut-of-the-phase. Cut strength is
/// normalised by total within-community edge weight so it is comparable across
/// communities of different sizes.
///
/// ### Params
///
/// * `outgoing` - CSR outgoing-neighbour graph.
/// * `cells` - Global cell indices forming the subgraph.
///
/// ### Returns
///
/// `Some((cut_strength, partition_a, partition_b))` where `partition_a` and
/// `partition_b` are global cell indices on either side of the minimum cut.
/// Returns `None` if `cells.len() < 2`.
fn stoer_wagner(
    outgoing: &CompressedSparseData2<f32, f32>,
    cells: &[usize],
) -> Option<(f64, Vec<usize>, Vec<usize>)> {
    let n = cells.len();
    if n < 2 {
        return None;
    }

    // map global cell index → local index in [0, n).
    let mut local_of_global: Vec<i32> = vec![-1; outgoing.shape.0];
    for (local, &global) in cells.iter().enumerate() {
        local_of_global[global] = local as i32;
    }

    // Build symmetric weight matrix on the local index space.
    // weight[i][j] = sum of outgoing weights in either direction.
    let mut weight = vec![vec![0.0f64; n]; n];
    let mut total_weight = 0.0f64;
    for (local_i, &global_i) in cells.iter().enumerate() {
        let start = outgoing.indptr[global_i] as usize;
        let end = outgoing.indptr[global_i + 1] as usize;
        for idx in start..end {
            let global_j = outgoing.indices[idx] as usize;
            let local_j = local_of_global[global_j];
            if local_j < 0 {
                continue;
            }
            let local_j = local_j as usize;
            if local_j == local_i {
                continue;
            }
            let w = outgoing.data[idx] as f64;
            weight[local_i][local_j] += w;
            weight[local_j][local_i] += w;
            total_weight += w;
        }
    }

    if total_weight <= 0.0 {
        // disconnected: arbitrary singleton split.
        let mut a = Vec::with_capacity(1);
        let mut b = Vec::with_capacity(n - 1);
        a.push(cells[0]);
        b.extend_from_slice(&cells[1..]);
        return Some((0.0, a, b));
    }

    // groups[v] = the original local indices currently merged into v.
    let mut groups: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut alive = vec![true; n];
    let mut best_cut = f64::INFINITY;
    let mut best_partition: Vec<usize> = Vec::new();

    for _ in 0..(n - 1) {
        // maximum-adjacency ordering on alive vertices.
        let alive_set: Vec<usize> = (0..n).filter(|&i| alive[i]).collect();
        let m = alive_set.len();
        if m < 2 {
            break;
        }
        let mut in_a = vec![false; n];
        let mut weights_to_a = vec![0.0f64; n];

        // start from the first alive vertex.
        let mut order: Vec<usize> = Vec::with_capacity(m);
        let start = alive_set[0];
        in_a[start] = true;
        order.push(start);
        for &v in &alive_set {
            if v != start {
                weights_to_a[v] = weight[start][v];
            }
        }

        for _ in 1..m {
            // pick the alive, not-in-A vertex with the maximum weight to A.
            let mut best = usize::MAX;
            let mut best_w = f64::NEG_INFINITY;
            for &v in &alive_set {
                if !in_a[v] && weights_to_a[v] > best_w {
                    best = v;
                    best_w = weights_to_a[v];
                }
            }
            if best == usize::MAX {
                break;
            }
            in_a[best] = true;
            order.push(best);
            for &v in &alive_set {
                if !in_a[v] {
                    weights_to_a[v] += weight[best][v];
                }
            }
        }

        if order.len() < 2 {
            break;
        }

        // last vertex added defines the cut-of-the-phase.
        let last = order[order.len() - 1];
        let prev = order[order.len() - 2];
        // weight of cut separating `last`
        let cut_w = weights_to_a[last];

        if cut_w < best_cut {
            best_cut = cut_w;
            best_partition = groups[last].clone();
        }

        // merge `last` into `prev`.
        let last_group = std::mem::take(&mut groups[last]);
        groups[prev].extend(last_group);
        alive[last] = false;
        // update weight matrix: weight[prev][v] += weight[last][v].
        for v in 0..n {
            if v != prev && v != last && alive[v] {
                weight[prev][v] += weight[last][v];
                weight[v][prev] += weight[v][last];
            }
        }
    }

    if best_partition.is_empty() {
        return None;
    }

    let strength = best_cut / total_weight;
    let small_set: FxHashSet<usize> = best_partition.into_iter().collect();
    let mut a = Vec::with_capacity(small_set.len());
    let mut b = Vec::with_capacity(n - small_set.len());
    for (local, &global) in cells.iter().enumerate() {
        if small_set.contains(&local) {
            a.push(global);
        } else {
            b.push(global);
        }
    }
    Some((strength, a, b))
}

/// Identify communities that are too small by cell count or UMI total.
///
/// A community is flagged if it is below `lowest_metacell_size`, below
/// `min_metacell_size` with insufficient UMIs, or below `min_metacell_umis`
/// with insufficient cells.
///
/// ### Params
///
/// * `partition_of_cells` - Current assignment vector.
/// * `cell_umis` - Per-cell UMI counts.
/// * `min_metacell_umis` - Minimum acceptable UMI total per metacell.
/// * `max_metacell_umis` - Upper UMI bound; used in the size threshold check.
/// * `min_metacell_size` - Minimum acceptable cell count per metacell.
/// * `max_metacell_size` - Upper cell-count bound; used in the UMI threshold
///   check.
/// * `lowest_metacell_size`- Hard minimum; communities below this are always
///   flagged.
///
/// ### Returns
///
/// A tuple `(small_community_ids, total_cell_count, total_umis)`.
fn find_small_communities(
    partition_of_cells: &[i32],
    cell_umis: &[f32],
    min_metacell_umis: f64,
    max_metacell_umis: f64,
    min_metacell_size: usize,
    max_metacell_size: usize,
    lowest_metacell_size: usize,
) -> (Vec<usize>, usize, f64) {
    let n_partitions = max_partition_id(partition_of_cells) + 1;
    let mut sizes = vec![0usize; n_partitions];
    let mut umis = vec![0.0f64; n_partitions];
    for (i, &p) in partition_of_cells.iter().enumerate() {
        if p >= 0 {
            sizes[p as usize] += 1;
            umis[p as usize] += cell_umis[i] as f64;
        }
    }

    let mut small = Vec::new();
    let mut total_size = 0usize;
    let mut total_umis = 0.0f64;
    for p in 0..n_partitions {
        let s = sizes[p];
        let u = umis[p];
        if s == 0 {
            continue;
        }
        let is_tiny = s < lowest_metacell_size;
        let too_few = s < min_metacell_size && 2.0 * u < max_metacell_umis;
        let too_small_umis = u < min_metacell_umis && 2 * s < max_metacell_size;
        if is_tiny || too_few || too_small_umis {
            small.push(p);
            total_size += s;
            total_umis += u;
        }
    }

    (small, total_size, total_umis)
}

/// Demote all cells in `cancelled` to outliers and renumber surviving
/// communities to densely fill `[0, kept_count)`.
///
/// ### Params
///
/// * `partition_of_cells` - Current assignment; modified in place.
/// * `cancelled` - Community IDs to dissolve.
///
/// ### Returns
///
/// The number of surviving communities after cancellation.
fn cancel_communities(partition_of_cells: &mut [i32], cancelled: &[usize]) -> usize {
    let cancelled_set: FxHashSet<usize> = cancelled.iter().copied().collect();
    let n_partitions = max_partition_id(partition_of_cells) + 1;
    let mut new_id = vec![-1i32; n_partitions];
    let mut next = 0i32;
    for p in 0..n_partitions {
        if !cancelled_set.contains(&p) {
            new_id[p] = next;
            next += 1;
        }
    }
    for v in partition_of_cells.iter_mut() {
        if *v >= 0 {
            let nid = new_id[*v as usize];
            *v = nid; // either remapped or -1 if cancelled
        }
    }
    next as usize
}

/// Renumber partition IDs to fill `[0, k)` with no gaps.
///
/// ### Params
///
/// * `partition_of_cells` - Current assignment; modified in place.
fn compact_partition_ids(partition_of_cells: &mut [i32]) {
    let n_partitions = max_partition_id(partition_of_cells) + 1;
    let mut seen = vec![false; n_partitions];
    for &p in partition_of_cells.iter() {
        if p >= 0 {
            seen[p as usize] = true;
        }
    }
    let mut remap = vec![-1i32; n_partitions];
    let mut next = 0i32;
    for (p, &s) in seen.iter().enumerate() {
        if s {
            remap[p] = next;
            next += 1;
        }
    }
    for v in partition_of_cells.iter_mut() {
        if *v >= 0 {
            *v = remap[*v as usize];
        }
    }
}

/// Return the largest non-negative partition ID, or `0` if all entries are
/// `-1`.
///
/// ### Params
///
/// * `partition_of_cells` - Current assignment vector.
///
/// ### Returns
///
/// The maximum ID as `usize`.
fn max_partition_id(partition_of_cells: &[i32]) -> usize {
    partition_of_cells
        .iter()
        .copied()
        .filter(|&p| p >= 0)
        .max()
        .unwrap_or(-1)
        .max(-1) as usize
}

/// Transpose `outgoing` and re-flag the result as CSR so that row `i` lists
/// cell `i`'s incoming neighbours.
///
/// ### Params
///
/// * `outgoing` - CSR outgoing-neighbour graph.
///
/// ### Returns
///
/// A new CSR matrix of the same shape representing incoming neighbours.
pub fn make_incoming_view(
    outgoing: &CompressedSparseData2<f32, f32>,
) -> CompressedSparseData2<f32, f32> {
    let t = transpose_sparse(outgoing);
    CompressedSparseData2 {
        data: t.data,
        indices: t.indices,
        indptr: t.indptr,
        cs_type: CompressedSparseFormat::Csr,
        data_2: t.data_2,
        shape: t.shape,
    }
}

//////////
// Main //
//////////

/// Compute candidate metacells. Top-level entry point matching upstream's
/// `compute_candidate_metacells`.
///
/// ### Params
///
/// * `outgoing` - CSR weighted graph (cell-cell similarity KNN).
/// * `incoming` - Same matrix transposed and re-flagged as CSR.
/// * `cell_umis` - Per-cell UMI counts as `f32` mass.
/// * `params` - Top-level metacells parameters.
/// * `random_seed` - Master RNG seed.
///
/// ### Returns
///
/// `partition_of_cells` of length `n`, with every entry `>= 0` and IDs in
/// `[0, n_partitions)`.
pub fn compute_candidate_metacells(
    outgoing: &CompressedSparseData2<f32, f32>,
    incoming: &CompressedSparseData2<f32, f32>,
    cell_umis: &[f32],
    params: &MetacellsParams,
    random_seed: u64,
) -> Vec<i32> {
    let n = outgoing.shape.0;
    assert_eq!(cell_umis.len(), n, "cell_umis length must match graph");
    assert_eq!(incoming.shape.0, n);

    let lowest_metacell_size = params.min_metacell_size;
    let target_size = params.target_metacell_size;
    let target_umis = params.target_metacell_umis as f64;
    let max_merge: f64 = params.partition.max_merge_size_factor;
    let min_split: f64 = params.partition.min_split_size_factor;

    let min_metacell_umis = (target_umis * max_merge).floor();
    let max_metacell_umis = (target_umis * min_split).ceil() - 1.0;
    let min_metacell_size: usize =
        lowest_metacell_size.max((target_size as f64 * max_merge) as usize);
    let max_metacell_size_f =
        ((target_size as f64 * min_split).ceil() - 1.0).max((min_metacell_size + 1) as f64);
    let mut max_metacell_size = max_metacell_size_f;

    let total_umis: f64 = cell_umis.iter().map(|&u| u as f64).sum();

    // Phase 0: choose initial seeds.
    let initial_seeds_count = seeds_count_for(
        n,
        total_umis,
        min_metacell_size,
        max_metacell_size as usize,
        min_metacell_umis,
        max_metacell_umis,
    );

    let mut partition_of_cells = vec![-1i32; n];
    let _seeds_count = choose_seeds(
        outgoing,
        incoming,
        &mut partition_of_cells,
        initial_seeds_count,
        params.partition.min_seed_size_quantile,
        params.partition.max_seed_size_quantile,
        random_seed,
    );

    let cooldown_pass = params.partition.cooldown_pass;
    let cooldown_phase = params.partition.cooldown_phase;
    let mut cold_temperature = 1.0 - cooldown_pass;

    let mut atomic_candidates: BTreeSet<Vec<usize>> = BTreeSet::new();
    let mut hot_communities: Vec<usize> = (0..initial_seeds_count).collect();

    let mut old_score = 1e9;
    let mut old_partition: Option<Vec<i32>> = None;
    let mut old_small_nodes_count = n;

    loop {
        // Reduce phase: optimise + cut/split until stable.
        let (new_cold_temperature, score) = reduce_communities(
            outgoing,
            incoming,
            &mut partition_of_cells,
            &mut hot_communities,
            cell_umis,
            min_metacell_umis,
            max_metacell_umis,
            target_umis,
            min_metacell_size,
            target_size,
            &mut max_metacell_size,
            params,
            cold_temperature,
            &mut atomic_candidates,
            random_seed,
        );
        cold_temperature = new_cold_temperature * (1.0 - cooldown_phase);

        // Look for small communities to dissolve.
        let (small_communities, small_nodes_count, small_nodes_umis) = find_small_communities(
            &partition_of_cells,
            cell_umis,
            min_metacell_umis,
            max_metacell_umis,
            min_metacell_size,
            max_metacell_size as usize,
            lowest_metacell_size,
        );

        if small_communities.len() < 2 {
            break;
        }

        // Acceptance check: did this iteration actually improve things?
        if (old_small_nodes_count, old_score) <= (small_nodes_count, score) {
            // Revert to the previous snapshot.
            if let Some(prev) = old_partition.take() {
                partition_of_cells = prev;
            }
            break;
        }

        old_score = score;
        old_partition = Some(partition_of_cells.clone());
        old_small_nodes_count = small_nodes_count;
        let _ = small_nodes_umis;

        // Drop the small communities and reseed.
        let kept = cancel_communities(&mut partition_of_cells, &small_communities);

        let small_seeds_count = (small_communities.len() - 1).min(seeds_count_for(
            small_nodes_count,
            small_nodes_umis,
            min_metacell_size,
            max_metacell_size as usize,
            min_metacell_umis,
            max_metacell_umis,
        ));

        let max_seeds_count = kept + small_seeds_count;
        let new_total = choose_seeds(
            outgoing,
            incoming,
            &mut partition_of_cells,
            max_seeds_count,
            params.partition.min_seed_size_quantile,
            params.partition.max_seed_size_quantile,
            random_seed,
        );
        hot_communities = (kept..new_total).collect();
    }

    // Compact partition IDs so output is dense 0..k.
    compact_partition_ids(&mut partition_of_cells);
    partition_of_cells
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph(
        edges: &[(usize, usize, f32)],
        n: usize,
    ) -> (
        CompressedSparseData2<f32, f32>,
        CompressedSparseData2<f32, f32>,
    ) {
        let mut sorted: Vec<(usize, usize, f32)> = edges.to_vec();
        sorted.sort_unstable_by_key(|a| (a.0, a.1));
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0usize; n + 1];
        for &(r, c, v) in &sorted {
            data.push(v);
            indices.push(c);
            indptr[r + 1] += 1;
        }
        for i in 0..n {
            indptr[i + 1] += indptr[i];
        }
        let outgoing = CompressedSparseData2 {
            data,
            indices: indices.index_cast(),
            indptr: indptr.index_cast(),
            cs_type: CompressedSparseFormat::Csr,
            data_2: None,
            shape: (n, n),
        };
        let incoming = make_incoming_view(&outgoing);
        (outgoing, incoming)
    }

    /// The minimum cut lands on the weak bridge rather than shaving off a single node.
    #[test]
    fn stoer_wagner_separates_two_cliques() {
        // Two K3 cliques weakly linked by 0<->3.
        let edges = vec![
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (0, 2, 1.0),
            (2, 0, 1.0),
            (3, 4, 1.0),
            (4, 3, 1.0),
            (4, 5, 1.0),
            (5, 4, 1.0),
            (3, 5, 1.0),
            (5, 3, 1.0),
            (0, 3, 0.01),
            (3, 0, 0.01),
        ];
        let (out, _inc) = make_graph(&edges, 6);
        let cells: Vec<usize> = (0..6).collect();
        let (strength, a, b) = stoer_wagner(&out, &cells).expect("min cut");

        assert!(
            strength < 0.05,
            "weak bridge should give low strength, got {}",
            strength
        );
        assert_eq!(a.len() + b.len(), 6);
        let a_set: FxHashSet<usize> = a.into_iter().collect();
        let b_set: FxHashSet<usize> = b.into_iter().collect();
        let cluster1: FxHashSet<usize> = [0, 1, 2].into_iter().collect();
        let cluster2: FxHashSet<usize> = [3, 4, 5].into_iter().collect();
        assert!(
            (a_set == cluster1 && b_set == cluster2) || (a_set == cluster2 && b_set == cluster1),
            "should split exactly along the cluster boundary"
        );
    }

    /// A graph with no bridging edge reports zero cut strength instead of failing.
    #[test]
    fn stoer_wagner_disconnected_components() {
        // Two disconnected pairs.
        let edges = vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)];
        let (out, _inc) = make_graph(&edges, 4);
        let cells: Vec<usize> = (0..4).collect();
        let (strength, _a, _b) = stoer_wagner(&out, &cells).expect("min cut");
        assert!(strength < 1e-9, "disconnected should give zero strength");
    }

    /// Cancelled communities drop to -1 and the survivors are renumbered densely.
    #[test]
    fn cancel_communities_renumbers_correctly() {
        let mut p = vec![0i32, 1, 2, 0, 1, 2, 3, 3];
        let cancelled = vec![1, 3];
        let kept = cancel_communities(&mut p, &cancelled);
        assert_eq!(kept, 2);
        assert_eq!(p, vec![0, -1, 1, 0, -1, 1, -1, -1]);
    }

    /// Gaps in the partition IDs are closed in order while outliers stay at -1.
    #[test]
    fn compact_partition_ids_fills_gaps() {
        let mut p = vec![0i32, 5, -1, 5, 0, 2];
        compact_partition_ids(&mut p);
        assert_eq!(p, vec![0, 2, -1, 2, 0, 1]);
    }

    /// Only communities failing both the size and the UMI floor are flagged as small.
    #[test]
    fn find_small_communities_flags_below_threshold() {
        let p = vec![0i32, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2];
        let umis = vec![100.0f32; 11];
        let (small, total_size, _total_umis) = find_small_communities(
            &p, &umis, 50.0,  // min_metacell_umis
            500.0, // max_metacell_umis
            3,     // min_metacell_size
            10,    // max_metacell_size
            2,     // lowest_metacell_size
        );
        // Community 1 has 2 cells, 200 UMIs; min_size is 3, 2*200=400 < 500, so flagged.
        // Communities 0 and 2 are fine.
        assert_eq!(small, vec![1]);
        assert_eq!(total_size, 2);
    }
}
