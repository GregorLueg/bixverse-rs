//! Candidate metacell orchestration.
//!
//! Port of upstream MC2's `compute_candidate_metacells`
//! (`metacells/tools/candidates.py`). Wraps the SA partition optimiser
//! (`partition.rs`) in an outer loop that:
//!
//! 1. Optimises an initial seed partition.
//! 2. Splits oversized partitions (random multi-way) and cuts loosely-coupled
//!    partitions (min-cut), reseeds the affected cells, and re-optimises.
//!    Repeats until no further splits/cuts.
//! 3. Detects small partitions, cancels them, reseeds their cells, and
//!    re-optimises. Repeats until fewer than two small partitions remain or
//!    the result stops improving, in which case the previous one is kept.
//!
//! ### Min cut
//!
//! Upstream hands igraph `float32` capacities, which igraph ignores, so its
//! cut minimises the number of edges while the reported strength uses the
//! weights. This port does the same: the cut is an unweighted Stoer-Wagner
//! over the symmetrised community graph and the strength is upstream's mean
//! weight ratio. In practice that leaves connected communities alone unless
//! a cut with fewer edges than the minimum degree exists.
//!
//! ### Outputs
//!
//! Every cell ends up in a candidate, with dense IDs `0..n_partitions`.

use std::collections::BTreeSet;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::prelude::*;
use bixverse_core::core::math::sparse::transpose_sparse;

use super::params::MetacellsParams;
use super::partition::optimise_partitions;
use super::seeds::{choose_seeds, seeds_count_for};

////////////
// Consts //
////////////

/// Growth of the upper size bound per inner phase. Upstream's `increase_phase`.
const INCREASE_PHASE: f64 = 1.01;

/// Starting score of the outer loop, above any real score so the first pass
/// always counts as an improvement. Upstream's `old_score = 1e9`.
const INITIAL_OLD_SCORE: f64 = 1e9;

///////////
// Types //
///////////

/// Min-cut result on one community.
#[derive(Debug)]
pub(super) struct MinCut {
    /// Upstream's cut strength; `None` when either side has no internal edge.
    pub(super) strength: Option<f64>,
    /// Global cell indices of the first side (the merged Stoer-Wagner group).
    pub(super) first: Vec<usize>,
    /// Global cell indices of the second side.
    pub(super) second: Vec<usize>,
}

/// Outcome of a min-cut attempt on a single community.
#[derive(PartialEq, Eq, Debug)]
enum CutAction {
    /// The community was left unchanged.
    Unchanged,
    /// The community was split into two communities.
    Split,
    /// The small side was demoted to outliers; the large side remains.
    Cut,
}

/// Size and UMI bounds shared by the stages of one candidates run.
#[derive(Clone, Copy, Debug)]
struct Bounds {
    /// Lower UMI bound per metacell.
    min_umis: f64,
    /// Target UMIs per metacell.
    target_umis: f64,
    /// Upper UMI bound per metacell.
    max_umis: f64,
    /// Lower cell-count bound per metacell.
    min_size: usize,
    /// Target cell count per metacell.
    target_size: usize,
    /// Upper cell-count bound per metacell, before any `INCREASE_PHASE` growth.
    max_size: usize,
}

/////////////
// Helpers //
/////////////

/// Group cell indices by partition id, ascending within each group.
///
/// ### Params
///
/// * `partition_of_cells` - Current assignment; entries `< 0` are skipped.
/// * `n_partitions` - Number of partitions.
///
/// ### Returns
///
/// One cell list per partition id.
fn bucket_cells_by_partition(partition_of_cells: &[i32], n_partitions: usize) -> Vec<Vec<usize>> {
    let mut buckets = vec![Vec::new(); n_partitions];
    for (cell, &p) in partition_of_cells.iter().enumerate() {
        if p >= 0 {
            buckets[p as usize].push(cell);
        }
    }
    buckets
}

/// Inner phase loop (upstream `_reduce_communities`): alternate SA
/// optimisation and cut/split passes until neither changes anything.
///
/// The RNG is seeded once per call, so the splits of successive inner
/// iterations draw from one stream, as upstream's `np.random.seed` at the top
/// of the function. The upper size bound grows by `INCREASE_PHASE` per
/// iteration and is local to the call.
///
/// ### Params
///
/// * `outgoing`, `incoming` - CSR graph and incoming-neighbour transpose.
/// * `partition_of_cells` - Current assignment; modified in place.
/// * `hot_communities` - Communities eligible for SA moves.
/// * `cell_umis` - Per-cell mass.
/// * `bounds` - Size and UMI bounds.
/// * `params` - Top-level MC2 parameters.
/// * `cold_temperature` - SA temperature for frozen communities.
/// * `atomic_candidates` - Cell sets a min cut already left unchanged.
/// * `random_seed` - Master RNG seed.
///
/// ### Returns
///
/// `(cold_temperature, score)` after convergence; the temperature has been
/// cooled once per inner iteration.
#[allow(clippy::too_many_arguments)]
fn reduce_communities(
    outgoing: &CompressedSparseData2<f32, f32>,
    incoming: &CompressedSparseData2<f32, f32>,
    partition_of_cells: &mut [i32],
    mut hot_communities: Vec<usize>,
    cell_umis: &[f32],
    bounds: &Bounds,
    params: &MetacellsParams,
    mut cold_temperature: f64,
    atomic_candidates: &mut BTreeSet<Vec<usize>>,
    random_seed: u64,
) -> Result<(f64, f64), ScErrors> {
    let mut rng = StdRng::seed_from_u64(random_seed);
    let mut max_size = bounds.max_size as f64;
    loop {
        let n_partitions = count_partitions(partition_of_cells);
        let mut hot_mask = vec![false; n_partitions];
        for &p in &hot_communities {
            if p < n_partitions {
                hot_mask[p] = true;
            }
        }

        let score = optimise_partitions(
            outgoing,
            incoming,
            cell_umis,
            partition_of_cells,
            bounds.min_umis,
            bounds.target_umis,
            bounds.max_umis,
            bounds.min_size as f64,
            bounds.target_size as f64,
            max_size,
            &params.partition,
            cold_temperature,
            hot_mask,
            random_seed,
        );

        cold_temperature *= 1.0 - params.partition.cooldown_phase;
        max_size *= INCREASE_PHASE;

        let (split_count, cut_count, new_hot) = cut_split_communities(
            outgoing,
            partition_of_cells,
            cell_umis,
            bounds,
            max_size,
            params,
            atomic_candidates,
            &mut rng,
        );
        hot_communities = new_hot;

        if split_count + cut_count == 0 {
            return Ok((cold_temperature, score));
        }

        let mut max_seeds_count = count_partitions(partition_of_cells);
        if cut_count > 0 {
            // a slot for the cut cells to form a community of their own
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
        )?;
    }
}

/// Run one round of random splits (phase A) and min-cut splitting (phase B),
/// as upstream's `_cut_split_communities`.
///
/// Phase A splits each community whose size or UMI total is above the current
/// maximum into random parts, keeping one random cell per part and demoting
/// the rest to `-1`. If any split happened, phase B is skipped.
///
/// Phase B min-cuts each community not already known to be atomic.
///
/// ### Params
///
/// * `outgoing` - CSR outgoing-neighbour graph.
/// * `partition_of_cells` - Current assignment; modified in place.
/// * `cell_umis` - Per-cell mass.
/// * `bounds` - Size and UMI bounds.
/// * `max_size` - Current, grown upper size bound.
/// * `params` - Top-level MC2 parameters.
/// * `atomic_candidates` - Memoisation set; cell sets already tried are
///   skipped.
/// * `rng` - The reduce call's RNG stream.
///
/// ### Returns
///
/// `(split_count, cut_count, new_hot_communities)`.
#[allow(clippy::too_many_arguments)]
fn cut_split_communities(
    outgoing: &CompressedSparseData2<f32, f32>,
    partition_of_cells: &mut [i32],
    cell_umis: &[f32],
    bounds: &Bounds,
    max_size: f64,
    params: &MetacellsParams,
    atomic_candidates: &mut BTreeSet<Vec<usize>>,
    rng: &mut StdRng,
) -> (usize, usize, Vec<usize>) {
    let n_partitions = count_partitions(partition_of_cells);
    let mut split_count = 0usize;
    let mut cut_count = 0usize;
    let mut next_new = n_partitions;
    let mut hot_communities: Vec<usize> = Vec::new();

    // Buckets are taken once: both phases only rewrite cells of the community
    // being visited, to ids >= n_partitions or -1, so later buckets stay valid.
    let buckets = bucket_cells_by_partition(partition_of_cells, n_partitions);

    // Phase A: random splits.
    for (community_index, cells) in buckets.iter().enumerate() {
        let community_size = cells.len();
        let community_umis: f64 = cells.iter().map(|&i| cell_umis[i] as f64).sum();

        let too_large_by_umis =
            community_umis > bounds.max_umis && community_size >= 3 * bounds.min_size;
        let too_large_by_size =
            community_size as f64 > max_size && community_umis >= 3.0 * bounds.min_umis;
        if !(too_large_by_umis || too_large_by_size) {
            continue;
        }

        // Python's round is half to even
        let by_umis = (community_umis / bounds.target_umis).round_ties_even() as usize;
        let by_size =
            (community_size as f64 / bounds.target_size as f64).round_ties_even() as usize;
        let split_parts_count = by_umis.min(by_size).max(2);

        let assignment = loop {
            let assignment: Vec<usize> = (0..community_size)
                .map(|_| rng.random_range(0..split_parts_count))
                .collect();
            if (0..split_parts_count).all(|s| assignment.contains(&s)) {
                break assignment;
            }
        };

        for &c in cells {
            partition_of_cells[c] = -1;
        }
        for split_index in 0..split_parts_count {
            let in_part: Vec<usize> = cells
                .iter()
                .zip(&assignment)
                .filter_map(|(&c, &s)| (s == split_index).then_some(c))
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

    // Phase B: min-cut. Buckets are ascending, so a bucket is its own memo key.
    // The cuts are pure functions of the graph and are solved in parallel; the
    // partition ids they hand out are then applied in community order.
    let cuts: Vec<Option<MinCut>> = buckets
        .par_iter()
        .map_init(
            || vec![-1_i32; outgoing.shape.0],
            |local_of_global, cells| {
                if cells.len() < 2 || atomic_candidates.contains(cells) {
                    return None;
                }
                Some(min_cut(outgoing, cells, local_of_global))
            },
        )
        .collect();

    for ((community_index, cells), cut) in buckets.iter().enumerate().zip(cuts) {
        let Some(cut) = cut else {
            continue;
        };

        match min_cut_community(
            partition_of_cells,
            cut,
            params.partition.max_split_min_cut_strength,
            params.partition.min_cut_seed_cells,
            params.must_complete_cover,
            next_new as i32,
        ) {
            CutAction::Unchanged => {
                atomic_candidates.insert(cells.to_vec());
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

/// Apply a min cut to one community, as upstream's `_min_cut_community`.
///
/// A cut with no strength (one side without internal edges) or with strength
/// above `max_split_min_cut_strength` leaves the community alone. Otherwise,
/// if the smaller side has at least `min_cut_seed_cells` cells, the second
/// side becomes a new community (split). Failing that the small side is
/// demoted to outliers (cut), unless `must_complete_cover` is set and the cut
/// is not a disconnection.
///
/// ### Params
///
/// * `partition_of_cells` - Current assignment; modified in place.
/// * `cut` - The [`min_cut`] result for the community.
/// * `max_split_min_cut_strength` - Strength above which the cut is rejected.
/// * `min_cut_seed_cells` - Smallest small side that is split off rather than
///   cut.
/// * `must_complete_cover` - If `true`, only disconnected cells are demoted.
/// * `new_community_index` - Partition ID for the second side on a split.
///
/// ### Returns
///
/// A [`CutAction`] describing the outcome.
fn min_cut_community(
    partition_of_cells: &mut [i32],
    cut: MinCut,
    max_split_min_cut_strength: f64,
    min_cut_seed_cells: usize,
    must_complete_cover: bool,
    new_community_index: i32,
) -> CutAction {
    let Some(strength) = cut.strength else {
        return CutAction::Unchanged;
    };
    if strength > max_split_min_cut_strength {
        return CutAction::Unchanged;
    }

    // upstream takes the first side only when it is strictly smaller
    let small = if cut.first.len() < cut.second.len() {
        &cut.first
    } else {
        &cut.second
    };

    if small.len() >= min_cut_seed_cells {
        for &c in &cut.second {
            partition_of_cells[c] = new_community_index;
        }
        return CutAction::Split;
    }

    if must_complete_cover && strength > 0.0 {
        return CutAction::Unchanged;
    }

    for &c in small {
        partition_of_cells[c] = -1;
    }
    CutAction::Cut
}

/// Upstream's min cut of the community graph `W + W^T` restricted to `cells`.
///
/// The cut itself is an unweighted Stoer-Wagner (every edge counts 1, see the
/// module docs). The strength is computed on the real weights:
/// `(cut / (|A| |B|)) / sqrt(mean_A * mean_B)` where `mean_X` is the internal
/// weight of side `X` over its pair count; `0` when the cut carries no weight
/// and `None` when either side has no internal weight.
///
/// ### Params
///
/// * `outgoing` - CSR outgoing-neighbour graph.
/// * `cells` - Global cell indices forming the subgraph, at least two.
/// * `local_of_global` - Scratch of length `outgoing.shape.0`, all `-1` on
///   entry and restored to all `-1` before return.
///
/// ### Returns
///
/// The [`MinCut`].
pub(super) fn min_cut(
    outgoing: &CompressedSparseData2<f32, f32>,
    cells: &[usize],
    local_of_global: &mut [i32],
) -> MinCut {
    let n = cells.len();
    debug_assert!(n >= 2);

    for (local, &global) in cells.iter().enumerate() {
        local_of_global[global] = local as i32;
    }

    // symmetric local weights, flat row-major n x n
    let mut weight = vec![0.0f64; n * n];
    for (local_i, &global_i) in cells.iter().enumerate() {
        let start = outgoing.indptr[global_i] as usize;
        let end = outgoing.indptr[global_i + 1] as usize;
        for idx in start..end {
            let local_j = local_of_global[outgoing.indices[idx] as usize];
            if local_j < 0 || local_j as usize == local_i {
                continue;
            }
            let local_j = local_j as usize;
            let w = outgoing.data[idx] as f64;
            weight[local_i * n + local_j] += w;
            weight[local_j * n + local_i] += w;
        }
    }
    for &global in cells {
        local_of_global[global] = -1;
    }

    let edges: Vec<f64> = weight
        .iter()
        .map(|&w| if w > 0.0 { 1.0 } else { 0.0 })
        .collect();
    let first_local = stoer_wagner(edges, n);

    let mut is_first = vec![false; n];
    for &l in &first_local {
        is_first[l] = true;
    }
    let (mut cut_w, mut first_w, mut second_w) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        for j in 0..i {
            let w = weight[i * n + j];
            match (is_first[i], is_first[j]) {
                (true, true) => first_w += w,
                (false, false) => second_w += w,
                _ => cut_w += w,
            }
        }
    }

    let first_size = first_local.len() as f64;
    let second_size = (n - first_local.len()) as f64;
    let strength = if cut_w == 0.0 {
        Some(0.0)
    } else if first_w > 0.0 && second_w > 0.0 {
        let cut_mean = cut_w / (first_size * second_size);
        let first_mean = first_w / (first_size * (first_size - 1.0) / 2.0);
        let second_mean = second_w / (second_size * (second_size - 1.0) / 2.0);
        Some(cut_mean / (first_mean * second_mean).sqrt())
    } else {
        None
    };

    let mut first = Vec::with_capacity(first_local.len());
    let mut second = Vec::with_capacity(n - first_local.len());
    for (local, &global) in cells.iter().enumerate() {
        if is_first[local] {
            first.push(global);
        } else {
            second.push(global);
        }
    }

    MinCut {
        strength,
        first,
        second,
    }
}

/// Stoer-Wagner global minimum cut.
///
/// Runs `n - 1` maximum-adjacency phases, merging the last two vertices of
/// each, and keeps the lightest cut-of-the-phase. `O(n^3)`; communities are
/// metacell sized, so this stays small.
///
/// ### Params
///
/// * `weight` - Symmetric flat row-major `n × n` capacities; consumed as
///   merge scratch.
/// * `n` - Vertex count, at least two.
///
/// ### Returns
///
/// The local vertex indices on one side of the minimum cut.
fn stoer_wagner(mut weight: Vec<f64>, n: usize) -> Vec<usize> {
    let mut groups: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut alive: Vec<usize> = (0..n).collect();
    let mut best_cut = f64::INFINITY;
    let mut best_group: Vec<usize> = Vec::new();

    let mut in_a = vec![false; n];
    let mut weights_to_a = vec![0.0f64; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);

    while alive.len() > 1 {
        for &v in &alive {
            in_a[v] = false;
            weights_to_a[v] = 0.0;
        }
        order.clear();

        for _ in 0..alive.len() {
            // the alive vertex outside A most tightly connected to A
            let mut best = usize::MAX;
            let mut best_w = f64::NEG_INFINITY;
            for &v in &alive {
                if !in_a[v] && weights_to_a[v] > best_w {
                    best = v;
                    best_w = weights_to_a[v];
                }
            }
            in_a[best] = true;
            order.push(best);
            for &v in &alive {
                if !in_a[v] {
                    weights_to_a[v] += weight[best * n + v];
                }
            }
        }

        let last = order[order.len() - 1];
        let prev = order[order.len() - 2];
        if weights_to_a[last] < best_cut {
            best_cut = weights_to_a[last];
            best_group = groups[last].clone();
        }

        let last_group = std::mem::take(&mut groups[last]);
        groups[prev].extend(last_group);
        alive.retain(|&v| v != last);
        for &v in &alive {
            if v != prev {
                let merged = weight[prev * n + v] + weight[last * n + v];
                weight[prev * n + v] = merged;
                weight[v * n + prev] = merged;
            }
        }
    }

    best_group
}

/// Identify communities that are too small by cell count or UMI total, as
/// upstream's `_find_small_communities`.
///
/// ### Params
///
/// * `partition_of_cells` - Current assignment vector.
/// * `cell_umis` - Per-cell mass.
/// * `bounds` - Size and UMI bounds (the ungrown ones).
/// * `lowest_metacell_size` - Hard minimum; communities below this are always
///   flagged.
///
/// ### Returns
///
/// A tuple `(small_community_ids, total_cell_count, total_umis)`.
fn find_small_communities(
    partition_of_cells: &[i32],
    cell_umis: &[f32],
    bounds: &Bounds,
    lowest_metacell_size: usize,
) -> (Vec<usize>, usize, f64) {
    let n_partitions = count_partitions(partition_of_cells);
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
        let (s, u) = (sizes[p], umis[p]);
        let is_tiny = s < lowest_metacell_size;
        let too_few = s < bounds.min_size && 2.0 * u < bounds.max_umis;
        let too_small_umis = u < bounds.min_umis && 2 * s < bounds.max_size;
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
    let n_partitions = count_partitions(partition_of_cells);
    let mut is_cancelled = vec![false; n_partitions];
    for &c in cancelled {
        is_cancelled[c] = true;
    }
    let mut new_id = vec![-1i32; n_partitions];
    let mut next = 0i32;
    for p in 0..n_partitions {
        if !is_cancelled[p] {
            new_id[p] = next;
            next += 1;
        }
    }
    for v in partition_of_cells.iter_mut() {
        if *v >= 0 {
            *v = new_id[*v as usize];
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
    let n_partitions = count_partitions(partition_of_cells);
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

/// Number of partition IDs in use: the largest ID plus one.
///
/// ### Params
///
/// * `partition_of_cells` - Current assignment vector.
///
/// ### Returns
///
/// `max + 1`, or `0` when every entry is `-1`.
fn count_partitions(partition_of_cells: &[i32]) -> usize {
    (partition_of_cells.iter().copied().max().unwrap_or(-1) + 1).max(0) as usize
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
/// * `cell_umis` - Per-cell mass (UMIs, or cells per group when the "cells"
///   are metacells in divide and conquer).
/// * `params` - Top-level metacells parameters.
/// * `random_seed` - Master RNG seed.
///
/// ### Returns
///
/// `partition_of_cells` of length `n`, with every entry `>= 0` and IDs in
/// `[0, n_partitions)`.
///
/// ### Errors
///
/// * `Mc2InvalidParams` if the size or UMI bounds collapse.
/// * `Mc2UnreachableCells` from seeding.
pub fn compute_candidate_metacells(
    outgoing: &CompressedSparseData2<f32, f32>,
    incoming: &CompressedSparseData2<f32, f32>,
    cell_umis: &[f32],
    params: &MetacellsParams,
    random_seed: u64,
) -> Result<Vec<i32>, ScErrors> {
    let n = outgoing.shape.0;
    assert_eq!(cell_umis.len(), n, "cell_umis length must match graph");
    assert_eq!(incoming.shape.0, n);

    let lowest_metacell_size = params.min_metacell_size;
    let target_size = params.target_metacell_size;
    let target_umis = params.target_metacell_umis as f64;
    let max_merge = params.partition.max_merge_size_factor;
    let min_split = params.partition.min_split_size_factor;

    let min_size = lowest_metacell_size.max((target_size as f64 * max_merge).floor() as usize);
    let bounds = Bounds {
        min_umis: (target_umis * max_merge).floor(),
        target_umis,
        max_umis: (target_umis * min_split).ceil() - 1.0,
        min_size,
        target_size,
        max_size: (((target_size as f64 * min_split).ceil() as usize).saturating_sub(1))
            .max(min_size + 1),
    };
    let umis_ordered = 0.0 < bounds.min_umis
        && bounds.min_umis < bounds.target_umis
        && bounds.target_umis < bounds.max_umis;
    let sizes_ordered =
        0 < bounds.min_size && bounds.min_size < target_size && target_size < bounds.max_size;
    if !(umis_ordered && sizes_ordered) {
        return Err(ScErrors::Mc2InvalidParams {
            reason: format!("candidate bounds must be ordered, got {bounds:?}"),
        });
    }

    let total_umis: f64 = cell_umis.iter().map(|&u| u as f64).sum();
    let initial_seeds_count = seeds_count_for(
        n,
        total_umis,
        bounds.min_size,
        bounds.max_size,
        bounds.min_umis,
        bounds.max_umis,
    );

    let mut partition_of_cells = vec![-1i32; n];
    let initial_seeds_count = choose_seeds(
        outgoing,
        incoming,
        &mut partition_of_cells,
        initial_seeds_count,
        params.partition.min_seed_size_quantile,
        params.partition.max_seed_size_quantile,
        random_seed,
    )?;

    let mut cold_temperature = 1.0 - params.partition.cooldown_pass;
    let mut atomic_candidates: BTreeSet<Vec<usize>> = BTreeSet::new();
    let mut hot_communities: Vec<usize> = (0..initial_seeds_count).collect();

    let mut old_score = INITIAL_OLD_SCORE;
    let mut old_partition: Option<Vec<i32>> = None;
    let mut old_small_nodes_count = n;

    loop {
        let (new_cold_temperature, score) = reduce_communities(
            outgoing,
            incoming,
            &mut partition_of_cells,
            hot_communities,
            cell_umis,
            &bounds,
            params,
            cold_temperature,
            &mut atomic_candidates,
            random_seed,
        )?;
        cold_temperature = new_cold_temperature;

        let (small_communities, small_nodes_count, small_nodes_umis) = find_small_communities(
            &partition_of_cells,
            cell_umis,
            &bounds,
            lowest_metacell_size,
        );

        if small_communities.len() < 2 {
            break;
        }

        if (old_small_nodes_count, old_score) <= (small_nodes_count, score) {
            if let Some(prev) = old_partition.take() {
                partition_of_cells = prev;
            }
            break;
        }

        old_score = score;
        old_partition = Some(partition_of_cells.clone());
        old_small_nodes_count = small_nodes_count;

        let kept = cancel_communities(&mut partition_of_cells, &small_communities);
        let small_seeds_count = (small_communities.len() - 1).min(seeds_count_for(
            small_nodes_count,
            small_nodes_umis,
            bounds.min_size,
            bounds.max_size,
            bounds.min_umis,
            bounds.max_umis,
        ));

        let seeds_count = choose_seeds(
            outgoing,
            incoming,
            &mut partition_of_cells,
            kept + small_seeds_count,
            params.partition.min_seed_size_quantile,
            params.partition.max_seed_size_quantile,
            random_seed,
        )?;
        hot_communities = (kept..seeds_count).collect();
    }

    compact_partition_ids(&mut partition_of_cells);
    Ok(partition_of_cells)
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

    /// Two dense blocks joined by a single edge: the edge-count cut is the
    /// bridge, and the weighted strength is low.
    #[test]
    fn test_min_cut_finds_single_bridge() {
        let mut edges = Vec::new();
        for block in [0..4, 4..8] {
            for i in block.clone() {
                for j in block.clone() {
                    if i != j {
                        edges.push((i, j, 1.0));
                    }
                }
            }
        }
        edges.push((0, 4, 0.01));
        let (out, _inc) = make_graph(&edges, 8);
        let cells: Vec<usize> = (0..8).collect();
        let cut = min_cut(&out, &cells, &mut [-1; 8]);
        let mut first = cut.first.clone();
        first.sort_unstable();
        assert!(first == vec![0, 1, 2, 3] || first == vec![4, 5, 6, 7]);
        assert!(cut.strength.unwrap() < 0.01);
    }

    /// A singleton side has no internal edges, so the strength is `None`.
    #[test]
    fn test_min_cut_singleton_has_no_strength() {
        // a triangle with a pendant node: the pendant has degree 1
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (3, 0, 1.0)];
        let (out, _inc) = make_graph(&edges, 4);
        let cut = min_cut(&out, &[0, 1, 2, 3], &mut [-1; 4]);
        assert_eq!(cut.first.len().min(cut.second.len()), 1);
        assert!(cut.strength.is_none());
    }

    /// Disconnected communities have zero strength.
    #[test]
    fn test_min_cut_disconnected_is_zero() {
        let edges = vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)];
        let (out, _inc) = make_graph(&edges, 4);
        let cut = min_cut(&out, &[0, 1, 2, 3], &mut [-1; 4]);
        assert_eq!(cut.strength, Some(0.0));
    }

    /// Cancelled communities drop to -1 and the survivors are renumbered densely.
    #[test]
    fn test_cancel_communities_renumbers() {
        let mut p = vec![0i32, 1, 2, 0, 1, 2, 3, 3];
        let kept = cancel_communities(&mut p, &[1, 3]);
        assert_eq!(kept, 2);
        assert_eq!(p, vec![0, -1, 1, 0, -1, 1, -1, -1]);
    }

    /// Gaps in the partition IDs are closed in order while outliers stay at -1.
    #[test]
    fn test_compact_partition_ids_fills_gaps() {
        let mut p = vec![0i32, 5, -1, 5, 0, 2];
        compact_partition_ids(&mut p);
        assert_eq!(p, vec![0, 2, -1, 2, 0, 1]);
    }

    /// Only communities failing the size and UMI floors are flagged as small.
    #[test]
    fn test_find_small_communities_flags_below_threshold() {
        let p = vec![0i32, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2];
        let umis = vec![100.0f32; 11];
        let bounds = Bounds {
            min_umis: 50.0,
            target_umis: 100.0,
            max_umis: 500.0,
            min_size: 3,
            target_size: 5,
            max_size: 10,
        };
        let (small, total_size, _) = find_small_communities(&p, &umis, &bounds, 2);
        // community 1: 2 cells, 200 UMIs; 2 < 3 and 2 * 200 < 500
        assert_eq!(small, vec![1]);
        assert_eq!(total_size, 2);
    }
}
