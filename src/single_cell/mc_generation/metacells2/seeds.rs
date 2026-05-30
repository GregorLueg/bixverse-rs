//! Initial seed assignment of cells to candidate metacells.
//!
//! ### Algorithm overview
//!
//! 1. **Greedy seeding from connected nodes.** For every unassigned node we
//!    track its number of unassigned incoming neighbours ("residual
//!    connectivity"). While candidates with `connectivity > 0` exist:
//!    select a seed via two-stage `nth_element` quantile filter, claim its
//!    top-`mean_seed_size` strongest unassigned incoming neighbours, decrement
//!    the residual connectivity of every node that pointed into the claimed
//!    set.
//!
//! 2. **Top-up by degree.** If we still need seeds but no connected
//!    candidates remain, fall back to picking high-degree (in × out)
//!    unassigned nodes as singleton seeds.
//!
//! 3. **Probabilistic completion (always runs).** Every remaining unassigned
//!    node samples a seed weighted by `edge_weight / seed_size` over its
//!    assigned neighbours (both directions). Iterate to fixed point. After
//!    this step every node is assigned — `must_complete_cover` does not
//!    gate this; it only governs whether deviant detection runs later.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Count unassigned incoming neighbours for each unassigned node.
///
/// Assigned nodes receive a count of `0` and are not candidates.
///
/// ### Params
///
/// * `incoming` - CSR matrix where row `i` lists cell `i`'s incoming
///   neighbours.
/// * `seed_of_cells` - Current seed assignments; `< 0` means unassigned.
///
/// ### Returns
///
/// A `Vec<i32>` of length `n` with the residual connectivity per node.
fn compute_initial_connectivity(
    incoming: &CompressedSparseData2<f32, f32>,
    seed_of_cells: &[i32],
) -> Vec<i32> {
    let n = seed_of_cells.len();
    let mut connectivity = vec![0i32; n];
    for node in 0..n {
        if seed_of_cells[node] >= 0 {
            continue;
        }
        let start = incoming.indptr[node] as usize;
        let end = incoming.indptr[node + 1] as usize;
        let mut c = 0i32;
        for idx in start..end {
            if seed_of_cells[incoming.indices[idx] as usize] < 0 {
                c += 1;
            }
        }
        connectivity[node] = c;
    }
    connectivity
}

/// Remove candidates with zero residual connectivity in place.
///
/// ### Params
///
/// * `candidates` - List of candidate node indices; modified in place.
/// * `connectivity` - Residual connectivity per node.
///
/// ### Returns
///
/// `true` if any candidates remain after filtering, `false` otherwise.
fn retain_connected(candidates: &mut Vec<usize>, connectivity: &[i32]) -> bool {
    candidates.retain(|&node| connectivity[node] > 0);
    !candidates.is_empty()
}

/// Pick a seed node from the `[min_q, max_q]` quantile band of residual
/// connectivity.
///
/// Reproduces upstream's two-`nth_element` band selection via a single
/// partial sort over an index vector. Samples uniformly at random within
/// the band.
///
/// ### Params
///
/// * `candidates` - Indices of connected candidate nodes.
/// * `connectivity` - Residual connectivity per node.
/// * `min_q` - Lower quantile bound `[0, 1]`.
/// * `max_q` - Upper quantile bound `[0, 1]`, `>= min_q`.
/// * `rng` - RNG used to draw uniformly from the band.
///
/// ### Returns
///
/// The chosen node index from `candidates`.
fn choose_seed_node(
    candidates: &[usize],
    connectivity: &[i32],
    min_q: f32,
    max_q: f32,
    rng: &mut StdRng,
) -> usize {
    let size = candidates.len();
    debug_assert!(size > 0);

    let min_rank = ((size as f32 - 1.0) * min_q).floor() as usize;
    let max_rank = ((size as f32 - 1.0) * max_q).ceil() as usize;
    debug_assert!(min_rank <= max_rank && max_rank < size);

    let mut positions: Vec<usize> = (0..size).collect();
    let key = |pos: &usize| connectivity[candidates[*pos]];

    // Partition so that positions[..=min_rank] are the `min_rank + 1`
    // smallest. After this, positions[min_rank + 1..] are >= positions[min_rank].
    positions.select_nth_unstable_by_key(min_rank, &key);

    // Now refine within positions[min_rank..] so the element at `max_rank`
    // is the (max_rank - min_rank)-th smallest of that suffix.
    if max_rank > min_rank {
        let (_lo, _mid, _hi) =
            positions[min_rank..].select_nth_unstable_by_key(max_rank - min_rank, &key);
    }

    let band_len = max_rank - min_rank + 1;
    let pick = rng.random_range(0..band_len);
    candidates[positions[min_rank + pick]]
}

/// Assign `seed_node` and its top-`mean_seed_size` strongest unassigned
/// incoming neighbours to `seed_id`.
///
/// Decrements residual connectivity for every node that had an outgoing edge
/// into any newly claimed node.
///
/// ### Params
///
/// * `outgoing` - CSR outgoing-neighbour graph.
/// * `incoming` - CSR incoming-neighbour graph.
/// * `seed_id` - Partition id to assign the claimed nodes.
/// * `seed_node` - Central node chosen as the seed.
/// * `mean_seed_size` - Maximum number of neighbours to claim alongside the
///   seed.
/// * `seed_of_cells` - Seed assignments; modified in place.
/// * `connectivity` - Residual connectivity per node; modified in place.
///
/// ### Returns
///
/// Nothing; `seed_of_cells` and `connectivity` are updated in place.
fn claim_seed(
    outgoing: &CompressedSparseData2<f32, f32>,
    incoming: &CompressedSparseData2<f32, f32>,
    seed_id: usize,
    seed_node: usize,
    mean_seed_size: usize,
    seed_of_cells: &mut [i32],
    connectivity: &mut [i32],
) {
    let in_start = incoming.indptr[seed_node] as usize;
    let in_end = incoming.indptr[seed_node + 1] as usize;

    // Filter incoming neighbours to those still unassigned.
    let mut neighbours: Vec<(u32, f32)> = (in_start..in_end)
        .filter_map(|idx| {
            let other = incoming.indices[idx];
            if seed_of_cells[other as usize] < 0 {
                Some((other, incoming.data[idx]))
            } else {
                None
            }
        })
        .collect();

    // Top-`mean_seed_size` by edge weight (descending). If we have fewer,
    // keep them all.
    if neighbours.len() > mean_seed_size {
        neighbours.select_nth_unstable_by(mean_seed_size, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        neighbours.truncate(mean_seed_size);
    }

    // Assign the seed node.
    debug_assert!(seed_of_cells[seed_node] < 0);
    seed_of_cells[seed_node] = seed_id as i32;
    connectivity[seed_node] = 0;

    // Assign the top neighbours.
    for &(node, _) in &neighbours {
        seed_of_cells[node as usize] = seed_id as i32;
        connectivity[node as usize] = 0;
    }

    // Decrement connectivity for every node that pointed into the claimed
    // set (one decrement per claimed node, per pointing-out edge).
    decrement_connectivity_for_outgoing_into(outgoing, seed_node, seed_of_cells, connectivity);
    for &(node, _) in &neighbours {
        decrement_connectivity_for_outgoing_into(
            outgoing,
            node as usize,
            seed_of_cells,
            connectivity,
        );
    }
}

/// Decrement residual connectivity for every unassigned outgoing neighbour
/// of `origin`.
///
/// Called after `origin` is claimed; its outgoing edges no longer count
/// toward the incoming connectivity of its still-unassigned neighbours.
///
/// ### Params
///
/// * `outgoing` - CSR outgoing-neighbour graph.
/// * `origin` - The newly claimed node whose edges are being removed.
/// * `seed_of_cells` - Current seed assignments.
/// * `connectivity` - Residual connectivity per node; decremented in place.
///
/// ### Returns
///
/// Nothing; `connectivity` is updated in place.
fn decrement_connectivity_for_outgoing_into(
    outgoing: &CompressedSparseData2<f32, f32>,
    origin: usize,
    seed_of_cells: &[i32],
    connectivity: &mut [i32],
) {
    let start = outgoing.indptr[origin] as usize;
    let end = outgoing.indptr[origin + 1] as usize;
    for idx in start..end {
        let other = outgoing.indices[idx] as usize;
        if seed_of_cells[other] < 0 {
            connectivity[other] = (connectivity[other] - 1).max(0);
        }
    }
}

/// Compute `(deg_in + 1) * (deg_out + 1)` for a node.
///
/// Used as a tie-breaking sort key in phase 2; matches upstream's
/// degree-product heuristic.
///
/// ### Params
///
/// * `outgoing` - CSR outgoing-neighbour graph.
/// * `incoming` - CSR incoming-neighbour graph.
/// * `node` - Node index to query.
///
/// ### Returns
///
/// The degree product as `u64`.
fn degree_product(
    outgoing: &CompressedSparseData2<f32, f32>,
    incoming: &CompressedSparseData2<f32, f32>,
    node: usize,
) -> u64 {
    let deg_out = (outgoing.indptr[node + 1] - outgoing.indptr[node]) as u64;
    let deg_in = (incoming.indptr[node + 1] - incoming.indptr[node]) as u64;
    (deg_in + 1) * (deg_out + 1)
}

/// Assign every remaining unassigned node to a seed by weighted sampling.
///
/// Each unassigned node sums `edge_weight / seed_size` contributions from
/// its assigned neighbours in both directions, then samples a seed proportional
/// to those weights. Iterates to fixed point; convergence is guaranteed because
/// at least one node is assigned per iteration.
///
/// ### Params
///
/// * `outgoing` - CSR outgoing-neighbour graph.
/// * `incoming` - CSR incoming-neighbour graph.
/// * `seed_of_cells` - Seed assignments; modified in place.
/// * `seeds_count` - Total number of seeds currently allocated.
/// * `rng` - RNG used for weighted sampling.
///
/// ### Returns
///
/// Nothing; `seed_of_cells` is fully populated on return.
fn complete_seeds(
    outgoing: &CompressedSparseData2<f32, f32>,
    incoming: &CompressedSparseData2<f32, f32>,
    seed_of_cells: &mut [i32],
    seeds_count: usize,
    rng: &mut StdRng,
) {
    if seeds_count == 0 {
        // Pathological: no seeds at all. Should not happen in practice
        // because phase 1 + phase 2 produce at least one seed for any
        // non-empty input.
        return;
    }

    let n = seed_of_cells.len();

    // seed_sizes counts current assignments per seed.
    let mut seed_sizes = vec![0i32; seeds_count];
    for &s in seed_of_cells.iter() {
        if s >= 0 {
            seed_sizes[s as usize] += 1;
        }
    }

    let mut weights = vec![0f64; seeds_count];

    // First pass: try to assign every node.
    let mut disconnected: Vec<usize> = Vec::new();
    for node in 0..n {
        if seed_of_cells[node] >= 0 {
            continue;
        }
        if !try_connect_node(
            outgoing,
            incoming,
            node,
            seed_of_cells,
            &mut seed_sizes,
            &mut weights,
            rng,
        ) {
            disconnected.push(node);
        }
    }

    // Subsequent passes: each iteration must shrink the disconnected list
    // strictly (an upstream invariant — if it didn't shrink, the graph
    // would have a fully-disconnected component, which the caller's KNN
    // construction precludes).
    while !disconnected.is_empty() {
        let prev_len = disconnected.len();
        let old = std::mem::take(&mut disconnected);
        for node in old {
            if !try_connect_node(
                outgoing,
                incoming,
                node,
                seed_of_cells,
                &mut seed_sizes,
                &mut weights,
                rng,
            ) {
                disconnected.push(node);
            }
        }
        debug_assert!(
            disconnected.len() < prev_len,
            "complete_seeds did not converge: {} disconnected nodes remain",
            disconnected.len()
        );
    }
}

/// Attempt to assign `node` to a seed by weighted sampling over its assigned
/// neighbours.
///
/// Accumulates `edge_weight / seed_size` per seed across both incoming and
/// outgoing neighbours, then samples proportionally. Falls back to the
/// highest-weight seed on floating-point overshoot.
///
/// ### Params
///
/// * `outgoing` - CSR outgoing-neighbour graph.
/// * `incoming` - CSR incoming-neighbour graph.
/// * `node` - The unassigned node to connect.
/// * `seed_of_cells` - Seed assignments; modified in place on success.
/// * `seed_sizes` - Current size of each seed; incremented in place on success.
/// * `weights` - Scratch buffer of length `seeds_count`; overwritten each call.
/// * `rng` - RNG used for weighted sampling.
///
/// ### Returns
///
/// `true` if `node` was assigned, `false` if no assigned neighbour was found.
fn try_connect_node(
    outgoing: &CompressedSparseData2<f32, f32>,
    incoming: &CompressedSparseData2<f32, f32>,
    node: usize,
    seed_of_cells: &mut [i32],
    seed_sizes: &mut [i32],
    weights: &mut [f64],
    rng: &mut StdRng,
) -> bool {
    weights.fill(0.0);
    let mut total = 0f64;

    for (mat, sign_label) in [(incoming, "in"), (outgoing, "out")] {
        let _ = sign_label; // distinguishing only for clarity; logic identical
        let start = mat.indptr[node] as usize;
        let end = mat.indptr[node + 1] as usize;
        for idx in start..end {
            let other = mat.indices[idx] as usize;
            let other_seed = seed_of_cells[other];
            if other_seed >= 0 {
                let w = (mat.data[idx] as f64) / (seed_sizes[other_seed as usize] as f64);
                weights[other_seed as usize] += w;
                total += w;
            }
        }
    }

    if total <= 0.0 {
        return false;
    }

    let mut pick = rng.random_range(0.0..total);
    for (seed_id, &w) in weights.iter().enumerate() {
        pick -= w;
        if pick <= 0.0 {
            seed_of_cells[node] = seed_id as i32;
            seed_sizes[seed_id] += 1;
            return true;
        }
    }
    // Floating-point edge case: pick may overshoot zero by < epsilon.
    // Fall through and assign to the highest-weighted seed.
    let (best, _) = weights
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("weights non-empty given seeds_count > 0");
    seed_of_cells[node] = best as i32;
    seed_sizes[best] += 1;
    true
}

//////////
// Main //
//////////

/// Compute the target seed count from the dataset's size and UMI bounds.
///
/// Mirrors `_seeds_count_for` in `candidates.py`: bounds the count from each
/// dimension (size, UMIs) by both the lower and upper extremes, then takes
/// the midpoint of whichever bound pair is feasible.
///
/// ### Params
///
/// * `total_size` - Number of cells.
/// * `total_umis` - Sum of `cell_umis`.
/// * `min_metacell_size`, `max_metacell_size` - Cell-count bounds per
///   metacell.
/// * `min_metacell_umis`, `max_metacell_umis` - UMI-count bounds per
///   metacell.
///
/// ### Returns
///
/// Seed count `>= 1`.
pub fn seeds_count_for(
    total_size: usize,
    total_umis: f64,
    min_metacell_size: usize,
    max_metacell_size: usize,
    min_metacell_umis: f64,
    max_metacell_umis: f64,
) -> usize {
    let ceil_div = |a: f64, b: f64| (a / b).ceil() as i64;

    let min_by_umis = ceil_div(total_umis, max_metacell_umis);
    let max_by_umis = ceil_div(total_umis, min_metacell_umis.max(1.0));
    let min_by_size = ceil_div(total_size as f64, max_metacell_size as f64);
    let max_by_size = ceil_div(total_size as f64, min_metacell_size.max(1) as f64);

    let count = if max_by_size < min_by_umis {
        (max_by_size + min_by_umis + 1) / 2
    } else if max_by_umis < min_by_size {
        (max_by_umis + min_by_size + 1) / 2
    } else {
        let lo = min_by_size.max(min_by_umis);
        let hi = max_by_size.min(max_by_umis);
        (lo + hi + 1) / 2
    };

    (count.max(1)) as usize
}

/// Choose seeds and complete the assignment so every node ends up in a seed.
///
/// On entry `seed_of_cells[i]` is either `>= 0` (cell already assigned to a
/// pre-existing seed by the caller — orchestrator uses this to preserve
/// kept communities across iterations) or `< 0` (unassigned). On exit every
/// entry is `>= 0` and `< returned_count`.
///
/// The returned count is the new total number of seeds, i.e. the number of
/// distinct partition ids in the output. It can be smaller than
/// `max_seeds_count` if phase 1 runs out of connected candidates and phase
/// 2 also runs out of unassigned cells.
///
/// ### Params
///
/// * `outgoing` - CSR `n × n` graph; row `i` lists cell `i`'s outgoing
///   neighbours and their edge weights.
/// * `incoming` - CSC of the same matrix (i.e. CSR layout where row `i`
///   lists cell `i`'s *incoming* neighbours). The caller is responsible for
///   producing this view.
/// * `seed_of_cells` - Mutable seed assignment; modified in place.
/// * `max_seeds_count` - Upper bound on the returned seed count.
/// * `min_seed_size_quantile`, `max_seed_size_quantile` - Quantile band
///   `[0, 1]` over residual connectivity from which the seed node is drawn
///   in phase 1.
/// * `random_seed` - RNG seed.
///
/// ### Panics
///
/// Panics if either matrix is non-square, if the matrix dimensions don't
/// match `seed_of_cells`, or if the quantile arguments are outside
/// `[0, 1]` and `min > max`.
pub fn choose_seeds(
    outgoing: &CompressedSparseData2<f32, f32>,
    incoming: &CompressedSparseData2<f32, f32>,
    seed_of_cells: &mut [i32],
    max_seeds_count: usize,
    min_seed_size_quantile: f32,
    max_seed_size_quantile: f32,
    random_seed: u64,
) -> usize {
    let n = seed_of_cells.len();
    assert_eq!(outgoing.shape, (n, n), "outgoing shape mismatch");
    assert_eq!(incoming.shape, (n, n), "incoming shape mismatch");
    assert!(
        (0.0..=1.0).contains(&min_seed_size_quantile)
            && (0.0..=1.0).contains(&max_seed_size_quantile)
            && min_seed_size_quantile <= max_seed_size_quantile,
        "invalid seed-size quantiles"
    );

    let given_seeds_count = seed_of_cells.iter().copied().max().unwrap_or(-1).max(-1) as i64 + 1;
    let mut seeds_count = given_seeds_count as usize;
    let mut rng = StdRng::seed_from_u64(random_seed);

    if seeds_count < max_seeds_count {
        // -- Phase 1: greedy seeding from connected nodes.
        let mut connectivity = compute_initial_connectivity(incoming, seed_of_cells);
        let mut candidates: Vec<usize> = (0..n).filter(|&i| seed_of_cells[i] < 0).collect();

        let unseeded = candidates.len();
        let to_create = max_seeds_count - seeds_count;
        // Upstream: ceil(unseeded / to_create). Guard the divisor.
        let mean_seed_size = if to_create == 0 {
            1
        } else {
            ((unseeded as f64) / (to_create as f64)).ceil() as usize
        }
        .max(1);

        while seeds_count < max_seeds_count && retain_connected(&mut candidates, &connectivity) {
            let seed_node = choose_seed_node(
                &candidates,
                &connectivity,
                min_seed_size_quantile,
                max_seed_size_quantile,
                &mut rng,
            );
            claim_seed(
                outgoing,
                incoming,
                seeds_count,
                seed_node,
                mean_seed_size,
                seed_of_cells,
                &mut connectivity,
            );
            seeds_count += 1;
        }

        // -- Phase 2: top-up by degree if still under the cap.
        if seeds_count < max_seeds_count {
            let mut leftover: Vec<usize> = (0..n).filter(|&i| seed_of_cells[i] < 0).collect();
            // Sort descending by (deg_in + 1) * (deg_out + 1).
            leftover.sort_unstable_by(|&a, &b| {
                let da = degree_product(outgoing, incoming, a);
                let db = degree_product(outgoing, incoming, b);
                db.cmp(&da)
            });
            for node in leftover.into_iter() {
                if seeds_count >= max_seeds_count {
                    break;
                }
                seed_of_cells[node] = seeds_count as i32;
                seeds_count += 1;
            }
        }
    }

    // -- Phase 3: probabilistic completion. Always runs.
    complete_seeds(outgoing, incoming, seed_of_cells, seeds_count, &mut rng);

    seeds_count
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::sparse::CompressedSparseFormat;

    fn make_graph(
        rows: Vec<usize>,
        cols: Vec<usize>,
        vals: Vec<f32>,
        n: usize,
    ) -> (
        CompressedSparseData2<f32, f32>,
        CompressedSparseData2<f32, f32>,
    ) {
        // Build outgoing CSR sorted by (row, col).
        let mut entries: Vec<(usize, usize, f32)> = rows
            .into_iter()
            .zip(cols)
            .zip(vals)
            .map(|((r, c), v)| (r, c, v))
            .collect();
        entries.sort_unstable_by_key(|a| (a.0, a.1));

        let mut data = Vec::with_capacity(entries.len());
        let mut indices = Vec::with_capacity(entries.len());
        let mut indptr = vec![0usize; n + 1];
        for &(r, c, v) in &entries {
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

        // Incoming = transpose of outgoing.
        let incoming = crate::core::math::sparse::transpose_sparse(&outgoing);
        // After transpose, layout flag flips. Re-flag as CSR (semantically:
        // row i lists incoming neighbours of node i).
        let incoming = CompressedSparseData2 {
            data: incoming.data,
            indices: incoming.indices,
            indptr: incoming.indptr,
            cs_type: CompressedSparseFormat::Csr,
            data_2: incoming.data_2,
            shape: incoming.shape,
        };

        (outgoing, incoming)
    }

    #[test]
    fn seeds_count_for_basic() {
        // Simple: total_size 1000, target metacell 100 cells / 10K UMIs;
        // bounds 50–150 cells, 5K–15K UMIs; total UMIs 100K.
        let n = seeds_count_for(1000, 100_000.0, 50, 150, 5_000.0, 15_000.0);
        // by_size: ceil(1000/150)=7 .. ceil(1000/50)=20
        // by_umis: ceil(100000/15000)=7 .. ceil(100000/5000)=20
        // overlap [7,20], midpoint round-up = 14
        assert_eq!(n, 14);
    }

    #[test]
    fn seeds_count_for_at_least_one() {
        assert!(seeds_count_for(1, 1.0, 1, 2, 1.0, 2.0) >= 1);
    }

    #[test]
    fn complete_assignment_after_phase3() {
        // Simple 4-node directed cycle with uniform weights:
        // 0 -> 1 -> 2 -> 3 -> 0, all weight 1.0.
        let (out, inc) = make_graph(
            vec![0, 1, 2, 3],
            vec![1, 2, 3, 0],
            vec![1.0, 1.0, 1.0, 1.0],
            4,
        );

        let mut seeds = vec![-1i32; 4];
        let count = choose_seeds(&out, &inc, &mut seeds, 2, 0.0, 1.0, 42);

        // Every node assigned, count <= 2.
        assert!(seeds.iter().all(|&s| s >= 0));
        assert!(
            seeds
                .iter()
                .copied()
                .all(|s| (0..count as i32).contains(&s))
        );
        assert!(seeds.iter().copied().all(|s| (s as usize) < count));
    }

    #[test]
    fn respects_partial_input() {
        // Pre-seed node 0 to partition 0; expect it to remain there.
        let (out, inc) = make_graph(
            vec![0, 1, 2, 3, 0, 1, 2, 3],
            vec![1, 2, 3, 0, 2, 3, 0, 1],
            vec![1.0; 8],
            4,
        );

        let mut seeds = vec![-1i32; 4];
        seeds[0] = 0;
        let count = choose_seeds(&out, &inc, &mut seeds, 2, 0.0, 1.0, 7);

        assert_eq!(seeds[0], 0);
        assert!(seeds.iter().all(|&s| s >= 0));
        assert!(count >= 1);
    }

    #[test]
    fn deterministic_under_same_seed() {
        let (out, inc) = make_graph(
            (0..6).flat_map(|i| [i, i]).collect(),
            (0..6)
                .flat_map(|i| vec![(i + 1) % 6, (i + 2) % 6])
                .collect(),
            vec![1.0; 12],
            6,
        );

        let mut a = vec![-1i32; 6];
        let mut b = vec![-1i32; 6];
        let ca = choose_seeds(&out, &inc, &mut a, 3, 0.0, 1.0, 99);
        let cb = choose_seeds(&out, &inc, &mut b, 3, 0.0, 1.0, 99);
        assert_eq!(ca, cb);
        assert_eq!(a, b);
    }
}
