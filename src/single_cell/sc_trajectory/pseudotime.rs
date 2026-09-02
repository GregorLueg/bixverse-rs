//! Geodesic pseudotime: the kNN graph Palantir walks, the repair that keeps it
//! connected, and the waypoint-perspective refinement loop.
//!
//! This is the memory-dominant half of the algorithm. The geodesic matrix is
//! `f64` and the weight matrix `f32`, both `n_waypoints x n_cells`, so at the
//! reference default of 1200 waypoints they cost `1200 * (8 + 4)` bytes, or
//! 14.4 KB per cell. `num_waypoints` is the knob.
//!
//! Both matrices are stored **row-major, waypoint by cell**. That is what the
//! Dijkstra fan-out writes naturally, and every consumer here is a per-cell
//! reduction over waypoints written as a cell-blocked loop with the waypoint
//! loop inner, so all reads stay contiguous and no transpose is ever needed.
//!
//! ### References
//!
//! Setty, et al., Nat. Biotechnol., 2019.

use rayon::prelude::*;

use crate::core::math::vector_helpers::pearson_correlation;
use crate::graph::graph_components::connected_components;
use crate::graph::shortest_paths::{dijkstra_from_source, multi_source_dijkstra};
use crate::prelude::*;

///////////////
// Constants //
///////////////

/// Cells processed per block in the fused waypoint reductions.
///
/// At `f64` a block spans 32 KB of each waypoint row, so the streamed slice and
/// the block accumulator both stay inside a typical 256 KB L2 while the waypoint
/// loop walks the matrix. Larger blocks spill; smaller ones pay the loop setup
/// once per waypoint for no gain.
const CELL_BLOCK: usize = 4096;

/// Silverman's rule-of-thumb multiplier for the geodesic kernel bandwidth.
///
/// `sdv = std(D) * 1.06 * n^(-1/5)` over every entry of the geodesic matrix,
/// exactly as the reference does.
const SILVERMAN_FACTOR: f64 = 1.06;

/// Exponent on the sample count in Silverman's rule.
const SILVERMAN_EXPONENT: f64 = -0.2;

/// Correlation between successive pseudotime vectors above which the refinement
/// is considered converged. The reference's value.
const PSEUDOTIME_CONVERGENCE_CORR: f64 = 0.9999;

/// Smallest refinement cap that actually runs a pass.
///
/// The loop counter starts at one to match the reference, so a cap of zero or
/// one leaves the range empty and the pseudotime is returned as the raw geodesic
/// row with no signal that nothing happened.
const MIN_MAX_ITERATIONS: usize = 2;

//////////////////////////
// Enums and structures //
//////////////////////////

/// Pseudotime together with the waypoint weights it was derived from.
pub struct PseudotimeResult {
    /// Pseudotime per cell, min-max scaled to `[0, 1]`.
    pub pseudotime: Vec<f32>,
    /// Column-normalised waypoint weights, row-major `n_waypoints x n_cells`.
    /// Each cell's weights sum to one, so `W^T B` is a convex combination.
    pub weights: Vec<f32>,
    /// Row count of `weights`.
    pub n_waypoints: usize,
    /// Column count of `weights`.
    pub n_cells: usize,
    /// Refinement passes actually run.
    pub iterations: usize,
    /// Whether the correlation criterion was met before the cap.
    pub converged: bool,
}

/////////////////////
// Graph construction //
/////////////////////

/// Build the symmetric CSR adjacency Palantir takes geodesics over.
///
/// An edge exists when either endpoint lists the other, matching the reference's
/// `directed=False` Dijkstra. Mutual edges are stored once per direction at a
/// single weight.
///
/// [crate::core::math::sparse::coo_to_csr] is deliberately not used here: it
/// sums duplicate `(row, col)` entries, so every mutual neighbour pair would
/// come out at twice its distance. Where a pair is reported in both directions
/// with slightly different rounding, the smaller weight wins, which makes the
/// result independent of input order.
///
/// ### Params
///
/// * `knn_indices` - kNN indices per cell, self excluded.
/// * `knn_distances` - kNN distances per cell, aligned with `knn_indices`.
///
/// ### Returns
///
/// A symmetric CSR adjacency, `n_cells` square, with no self loops.
pub fn build_symmetric_knn_graph(
    knn_indices: &[Vec<usize>],
    knn_distances: &[Vec<f32>],
) -> Result<CompressedSparseData2<f32>, BixverseErrors> {
    let n = knn_indices.len();
    if knn_distances.len() != n {
        return Err(BixverseErrors::DimensionMisMatchSparse {
            indices_len: n,
            data_len: knn_distances.len(),
        });
    }

    let mut edges: Vec<(u32, u32, f32)> =
        Vec::with_capacity(n * knn_indices.first().map_or(1, Vec::len).max(1));
    for (i, neighbours) in knn_indices.iter().enumerate() {
        if neighbours.len() != knn_distances[i].len() {
            return Err(BixverseErrors::DimensionMisMatchSparse {
                indices_len: neighbours.len(),
                data_len: knn_distances[i].len(),
            });
        }
        for (slot, &j) in neighbours.iter().enumerate() {
            if j == i {
                continue;
            }
            if j >= n {
                return Err(BixverseErrors::SliceIndexOutOfBounds { index: j, len: n });
            }
            let d = knn_distances[i][slot];
            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
            edges.push((lo as u32, hi as u32, d));
        }
    }

    edges.par_sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)).then(a.2.total_cmp(&b.2)));
    edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    Ok(edges_to_csr(n, &edges))
}

/// Scatter an undirected edge list into a symmetric CSR adjacency.
///
/// Rows come out sorted for free, without a per-row sort or a per-row
/// allocation. The trick is the scatter order: the `hi` side of every edge is
/// written first, filling each row `r` with its `lo < r` partners in ascending
/// order, and the `lo` side second, appending its `hi > r` partners also in
/// ascending order. Both halves are ascending because the input is sorted by
/// `(lo, hi)`, and the first half is entirely below the second because
/// `lo < hi`. This runs on every [connect_graph] pass, so the allocation it
/// avoids is the whole graph's worth.
///
/// ### Params
///
/// * `n` - Node count.
/// * `edges` - Deduplicated `(lo, hi, weight)` triples with `lo < hi`, sorted
///   ascending by `(lo, hi)`.
///
/// ### Returns
///
/// The CSR adjacency with both directions stored and every row ascending.
fn edges_to_csr(n: usize, edges: &[(u32, u32, f32)]) -> CompressedSparseData2<f32> {
    let mut degree = vec![0u32; n];
    for &(a, b, _) in edges {
        degree[a as usize] += 1;
        degree[b as usize] += 1;
    }

    let mut indptr = vec![0u32; n + 1];
    for i in 0..n {
        indptr[i + 1] = indptr[i] + degree[i];
    }

    let nnz = indptr[n] as usize;
    let mut indices = vec![0u32; nnz];
    let mut data = vec![0.0f32; nnz];
    let mut cursor: Vec<u32> = indptr[..n].to_vec();

    for &(a, b, w) in edges {
        let pb = cursor[b as usize] as usize;
        indices[pb] = a;
        data[pb] = w;
        cursor[b as usize] += 1;
    }
    for &(a, b, w) in edges {
        let pa = cursor[a as usize] as usize;
        indices[pa] = b;
        data[pa] = w;
        cursor[a as usize] += 1;
    }

    CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (n, n))
}

/// Reconnect a kNN graph so every cell is reachable from the start cell.
///
/// Repeatedly bridges the farthest reachable cell to its nearest unreachable
/// one and re-runs the reachability check, following the reference. Unreachable
/// cells otherwise leave infinities in the geodesic matrix, which propagate
/// through the bandwidth into every weight.
///
/// The Dijkstra is deliberately re-run from scratch after each added edge rather
/// than warm-started from the retained distances. That looks safe and is not:
/// once a second region is attached it can offer a shortcut into a region
/// attached earlier, so previously settled distances are not final.
///
/// Sequential by design. The loop body runs once per disconnected component,
/// which is normally zero and in practice a handful. It is nonetheless capped at
/// `n` passes: each pass rebuilds the whole CSR, so a bridging edge that fails
/// to attach anything, which non-finite weights used to cause, would otherwise
/// spin forever while `extra` grows without bound.
///
/// ### Params
///
/// * `graph` - Symmetric CSR adjacency from [build_symmetric_knn_graph].
/// * `data` - Multiscale components, cells by components, used to price the
///   bridging edges. Must have one row per node.
/// * `start_cell` - The cell everything must be reachable from.
///
/// ### Returns
///
/// The repaired graph and the number of edges added. The input is returned
/// untouched when it is already connected.
pub fn connect_graph(
    graph: &CompressedSparseData2<f32>,
    data: &[Vec<f32>],
    start_cell: usize,
) -> Result<(CompressedSparseData2<f32>, usize), BixverseErrors> {
    let n = graph.shape.0;
    if start_cell >= n {
        return Err(BixverseErrors::SliceIndexOutOfBounds {
            index: start_cell,
            len: n,
        });
    }
    if data.len() != n {
        return Err(BixverseErrors::DimensionMisMatchSparse {
            indices_len: data.len(),
            data_len: n,
        });
    }

    let (n_components, _) = connected_components(graph)?;
    if n_components <= 1 {
        return Ok((graph.clone(), 0));
    }

    let mut extra: Vec<(u32, u32, f32)> = Vec::new();
    let mut current = graph.clone();
    let mut dist = vec![0.0f64; n];

    // One bridge attaches at least one component, and there are at most `n` of
    // them, so a pass that does not shrink the unreachable set is a bug rather
    // than a slow case.
    for _ in 0..n {
        dijkstra_from_source(&current, start_cell, &mut dist)?;

        let unreachable: Vec<usize> = (0..n).filter(|&i| dist[i].is_infinite()).collect();
        if unreachable.is_empty() {
            return Ok((current, extra.len()));
        }

        // Farthest reachable cell, first index on ties.
        let mut anchor = start_cell;
        let mut best = f64::NEG_INFINITY;
        for (i, &d) in dist.iter().enumerate() {
            if d.is_finite() && d > best {
                best = d;
                anchor = i;
            }
        }

        let (target, weight) = unreachable
            .par_iter()
            .map(|&u| {
                let d: f32 = data[u]
                    .iter()
                    .zip(data[anchor].iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                (u, d.sqrt())
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
            .ok_or(BixverseErrors::PalantirDisconnectedGraph {
                n_unreachable: unreachable.len(),
                repairs: extra.len(),
            })?;

        let (lo, hi) = if anchor < target {
            (anchor as u32, target as u32)
        } else {
            (target as u32, anchor as u32)
        };
        extra.push((lo, hi, weight));
        current = rebuild_with_extra(graph, &extra);
    }

    dijkstra_from_source(&current, start_cell, &mut dist)?;
    Err(BixverseErrors::PalantirDisconnectedGraph {
        n_unreachable: dist.iter().filter(|d| d.is_infinite()).count(),
        repairs: extra.len(),
    })
}

/// Rebuild a CSR adjacency with additional undirected edges spliced in.
///
/// ### Params
///
/// * `graph` - The original symmetric CSR adjacency.
/// * `extra` - Additional `(lo, hi, weight)` edges with `lo < hi`.
///
/// ### Returns
///
/// A fresh CSR adjacency containing both edge sets, sorted and deduplicated
/// with the smaller weight winning, as [build_symmetric_knn_graph] does.
fn rebuild_with_extra(
    graph: &CompressedSparseData2<f32>,
    extra: &[(u32, u32, f32)],
) -> CompressedSparseData2<f32> {
    let n = graph.shape.0;
    let mut edges: Vec<(u32, u32, f32)> = Vec::with_capacity(graph.get_nnz() / 2 + extra.len());

    for i in 0..n {
        for idx in graph.indptr[i] as usize..graph.indptr[i + 1] as usize {
            let j = graph.indices[idx];
            if (i as u32) < j {
                edges.push((i as u32, j, graph.data[idx]));
            }
        }
    }
    edges.extend_from_slice(extra);

    edges.par_sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)).then(a.2.total_cmp(&b.2)));
    edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    edges_to_csr(n, &edges)
}

////////////////
// Pseudotime //
////////////////

/// Geodesic pseudotime by iterative waypoint refinement.
///
/// Every waypoint offers a signed perspective on each cell: cells later in
/// pseudotime sit at `t_w + d(w, c)`, earlier ones at `t_w - d(w, c)`. The new
/// pseudotime is the weighted average of those perspectives, with weights from
/// a Gaussian on the geodesic distance. The start cell's row is special-cased to
/// pure geodesic distance, which anchors the scale.
///
/// The inner loop is fused: the reference materialises a sign matrix, a
/// perspective matrix and their product, three full `n_waypoints x n_cells`
/// temporaries per pass. None of them are needed.
///
/// ### Params
///
/// * `graph` - Repaired symmetric CSR adjacency.
/// * `waypoints` - Waypoint cell indices. `waypoints[0]` must be `start_cell`;
///   the whole routine anchors its scale on geodesic row zero.
/// * `start_cell` - The cell pseudotime is measured from. Checked against
///   `waypoints[0]` rather than searched for, because the reference's
///   `waypoints.get_loc(start_cell)` is only well defined when the start cell is
///   in the sample at all.
/// * `max_iterations` - Iteration cap. The counter starts at one, so this
///   permits `max_iterations - 1` passes, matching the reference. Must be at
///   least `MIN_MAX_ITERATIONS`.
/// * `verbosity` - Controls progress reporting.
///
/// ### Returns
///
/// The [PseudotimeResult], or an error when the graph is still disconnected or
/// the pseudotime degenerates to a constant.
pub fn compute_pseudotime(
    graph: &CompressedSparseData2<f32>,
    waypoints: &[usize],
    start_cell: usize,
    max_iterations: usize,
    verbosity: Verbosity,
) -> Result<PseudotimeResult, BixverseErrors> {
    let n_cells = graph.shape.0;
    let n_wp = waypoints.len();

    if n_wp == 0 {
        return Err(BixverseErrors::InvalidArgument(
            "Palantir: no waypoints were supplied".to_string(),
        ));
    }
    if waypoints[0] != start_cell {
        return Err(BixverseErrors::PalantirStartCellNotFirstWaypoint {
            start_cell,
            first_waypoint: waypoints[0],
        });
    }
    if max_iterations < MIN_MAX_ITERATIONS {
        return Err(BixverseErrors::PalantirMaxIterationsTooSmall {
            max_iterations,
            minimum: MIN_MAX_ITERATIONS,
        });
    }

    if verbosity.normal_verbosity() {
        println!("Computing geodesic distances from {n_wp} waypoints...");
    }
    let geodesic = multi_source_dijkstra(graph, waypoints)?;

    let n_unreachable = count_unreachable_cells(&geodesic, n_wp, n_cells);
    if n_unreachable > 0 {
        return Err(BixverseErrors::PalantirUnreachableFromWaypoints { n_unreachable });
    }

    let weights = gaussian_waypoint_weights(&geodesic, n_wp, n_cells)?;

    if verbosity.normal_verbosity() {
        println!("Refining pseudotime (cap {max_iterations} iterations)...");
    }

    // The start cell is waypoints[0], so its geodesic row is row zero.
    let mut pseudotime: Vec<f64> = geodesic[..n_cells].to_vec();
    let mut next = vec![0.0f64; n_cells];
    let mut iterations = 0usize;
    let mut converged = false;

    for iteration in 1..max_iterations {
        refine_once(
            &geodesic,
            &weights,
            &pseudotime,
            waypoints,
            n_cells,
            &mut next,
        );

        let corr = pearson_correlation(&pseudotime, &next);
        iterations = iteration;
        std::mem::swap(&mut pseudotime, &mut next);

        if let Some(corr) = corr {
            if verbosity.detailed_verbosity() {
                println!("  correlation at iteration {iteration}: {corr:.6}");
            }
            if corr > PSEUDOTIME_CONVERGENCE_CORR {
                converged = true;
                break;
            }
        }
    }

    let pseudotime = normalise_pseudotime(pseudotime)?;

    Ok(PseudotimeResult {
        pseudotime,
        weights,
        n_waypoints: n_wp,
        n_cells,
        iterations,
        converged,
    })
}

/// Count the cells unreachable from at least one waypoint.
///
/// The naive `geodesic.iter().filter(|d| !d.is_finite()).count()` counts matrix
/// entries, which overstates the cell count by up to a factor of `n_wp`. Blocked
/// over cells with the waypoint loop inner so every read stays contiguous, as
/// everywhere else in this module.
///
/// ### Params
///
/// * `geodesic` - Row-major `n_wp x n_cells` geodesic distances.
/// * `n_wp` - Waypoint count.
/// * `n_cells` - Cell count.
///
/// ### Returns
///
/// The number of distinct cells with a non-finite distance to some waypoint.
fn count_unreachable_cells(geodesic: &[f64], n_wp: usize, n_cells: usize) -> usize {
    let mut bad = vec![false; n_cells];
    bad.par_chunks_mut(CELL_BLOCK)
        .enumerate()
        .for_each(|(block, out)| {
            let c0 = block * CELL_BLOCK;
            let len = out.len();
            for wp in 0..n_wp {
                let base = wp * n_cells + c0;
                let row = &geodesic[base..base + len];
                for k in 0..len {
                    out[k] |= !row[k].is_finite();
                }
            }
        });

    bad.par_iter().filter(|b| **b).count()
}

/// Population standard deviation, as a parallel two-pass reduction.
///
/// [crate::core::math::vector_helpers::standard_deviation] is sequential and
/// divides by `n - 1`. This is pointed at the geodesic matrix, the largest
/// buffer in the pipeline, from inside an otherwise fully parallel routine, and
/// the reference's Silverman rule uses the population form.
///
/// ### Params
///
/// * `x` - The values to reduce over.
///
/// ### Returns
///
/// The population standard deviation, or `0.0` for an empty slice.
fn population_standard_deviation(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let n = x.len() as f64;
    let mean = x.par_iter().sum::<f64>() / n;
    let variance = x.par_iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;

    variance.sqrt()
}

/// Gaussian waypoint weights with a Silverman bandwidth, normalised per cell.
///
/// The exponent is shifted by each cell's nearest waypoint before it is
/// exponentiated, the usual softmax stabilisation. Since the weights are
/// normalised per cell immediately afterwards, the shift cancels exactly and
/// changes nothing about the result, but it is not optional: Silverman's
/// bandwidth shrinks as `(n_waypoints * n_cells)^(-1/5)`, so on a sparsely
/// sampled manifold `exp(-0.5 * (d / sdv)^2)` underflows to zero for every
/// waypoint of some cell, and that cell's weights then sum to zero. The
/// reference divides by that zero and propagates `NaN` through the whole
/// refinement.
///
/// ### Params
///
/// * `geodesic` - Row-major `n_waypoints x n_cells` geodesic distances.
/// * `n_wp` - Waypoint count.
/// * `n_cells` - Cell count.
///
/// ### Returns
///
/// Row-major weights in the same layout, each cell's column summing to one, or
/// an error when the bandwidth degenerates.
fn gaussian_waypoint_weights(
    geodesic: &[f64],
    n_wp: usize,
    n_cells: usize,
) -> Result<Vec<f32>, BixverseErrors> {
    let sdv = population_standard_deviation(geodesic)
        * SILVERMAN_FACTOR
        * (n_wp as f64 * n_cells as f64).powf(SILVERMAN_EXPONENT);

    if !sdv.is_finite() || sdv <= 0.0 {
        return Err(BixverseErrors::PalantirDegeneratePseudotime {
            reason: "the Silverman bandwidth is zero or non-finite",
        });
    }
    let inv_two_var = 0.5 / (sdv * sdv);

    // Nearest waypoint per cell: block over cells, waypoints inner, so the
    // geodesic rows stream.
    let mut col_min = vec![f64::INFINITY; n_cells];
    col_min
        .par_chunks_mut(CELL_BLOCK)
        .enumerate()
        .for_each(|(block, out)| {
            let c0 = block * CELL_BLOCK;
            let len = out.len();
            for wp in 0..n_wp {
                let base = wp * n_cells + c0;
                let row = &geodesic[base..base + len];
                for k in 0..len {
                    if row[k] < out[k] {
                        out[k] = row[k];
                    }
                }
            }
        });

    let mut weights = vec![0.0f32; n_wp * n_cells];
    weights
        .par_chunks_mut(n_cells)
        .enumerate()
        .for_each(|(wp, row)| {
            let src = &geodesic[wp * n_cells..(wp + 1) * n_cells];
            for c in 0..n_cells {
                let d = src[c];
                let m = col_min[c];
                row[c] = (-(d * d - m * m) * inv_two_var).exp() as f32;
            }
        });

    let mut col_sum = vec![0.0f64; n_cells];
    col_sum
        .par_chunks_mut(CELL_BLOCK)
        .enumerate()
        .for_each(|(block, out)| {
            let c0 = block * CELL_BLOCK;
            let len = out.len();
            for wp in 0..n_wp {
                let base = wp * n_cells + c0;
                let row = &weights[base..base + len];
                for k in 0..len {
                    out[k] += row[k] as f64;
                }
            }
        });

    // The shift puts a weight of exactly one on each cell's nearest waypoint,
    // so a non-positive sum can now only come from a non-finite geodesic. The
    // reachability check above rules that out; this is belt and braces.
    if col_sum.par_iter().any(|s| !s.is_finite() || *s <= 0.0) {
        return Err(BixverseErrors::PalantirDegeneratePseudotime {
            reason: "a cell received zero total waypoint weight",
        });
    }

    weights.par_chunks_mut(n_cells).for_each(|row| {
        for (c, w) in row.iter_mut().enumerate() {
            *w = (*w as f64 / col_sum[c]) as f32;
        }
    });

    Ok(weights)
}

/// One fused pass of the waypoint-perspective refinement.
///
/// Parallelised over cell blocks rather than waypoints so each task owns a
/// disjoint output slice: no races, no thread-local accumulators, and reads stay
/// contiguous inside every waypoint row.
///
/// The reference computes its sign mask against the un-zeroed start-cell
/// pseudotime and only afterwards forces that row's sign to `+1` and its offset
/// to zero. Because the sign is forced unconditionally, the mask value there is
/// dead, so hoisting the start row out of the loop is exactly equivalent and
/// leaves the inner loop branch-free.
///
/// ### Params
///
/// * `geodesic` - Row-major `n_waypoints x n_cells` geodesic distances.
/// * `weights` - Row-major weights in the same layout.
/// * `pseudotime` - Current pseudotime per cell.
/// * `waypoints` - Waypoint cell indices.
/// * `n_cells` - Cell count.
/// * `out` - Output buffer of length `n_cells`, overwritten in full.
///
/// ### Returns
///
/// Nothing; `out` receives the refined pseudotime.
fn refine_once(
    geodesic: &[f64],
    weights: &[f32],
    pseudotime: &[f64],
    waypoints: &[usize],
    n_cells: usize,
    out: &mut [f64],
) {
    let n_wp = waypoints.len();
    let t_wp: Vec<f64> = waypoints.iter().map(|&w| pseudotime[w]).collect();

    out.par_chunks_mut(CELL_BLOCK)
        .enumerate()
        .for_each(|(block, out)| {
            let c0 = block * CELL_BLOCK;
            let len = out.len();

            // Start row: offset zero and sign forced to +1, so the term is D * W.
            let d_row = &geodesic[c0..c0 + len];
            let w_row = &weights[c0..c0 + len];
            for k in 0..len {
                out[k] = d_row[k] * w_row[k] as f64;
            }

            for wp in 1..n_wp {
                let base = wp * n_cells + c0;
                let d_row = &geodesic[base..base + len];
                let w_row = &weights[base..base + len];
                let t = t_wp[wp];
                for k in 0..len {
                    let sign = if pseudotime[c0 + k] < t { -1.0 } else { 1.0 };
                    out[k] += (d_row[k] * sign + t) * w_row[k] as f64;
                }
            }
        });
}

/// Min-max scale pseudotime into `[0, 1]`.
///
/// ### Params
///
/// * `pseudotime` - Raw pseudotime per cell.
///
/// ### Returns
///
/// The scaled pseudotime as `f32`, or an error when it is constant.
fn normalise_pseudotime(pseudotime: Vec<f64>) -> Result<Vec<f32>, BixverseErrors> {
    let lo = pseudotime
        .par_iter()
        .copied()
        .reduce(|| f64::INFINITY, f64::min);
    let hi = pseudotime
        .par_iter()
        .copied()
        .reduce(|| f64::NEG_INFINITY, f64::max);

    let range = hi - lo;
    if !range.is_finite() || range <= 0.0 {
        return Err(BixverseErrors::PalantirDegeneratePseudotime {
            reason: "pseudotime collapsed to a constant",
        });
    }

    Ok(pseudotime
        .par_iter()
        .map(|&t| ((t - lo) / range) as f32)
        .collect())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// kNN graph of a chain of `n` cells: two unit-spaced neighbours each, self excluded.
    fn chain_knn(n: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let mut indices = Vec::with_capacity(n);
        let mut distances = Vec::with_capacity(n);
        for i in 0..n {
            let mut idx = Vec::new();
            let mut dst = Vec::new();
            if i > 0 {
                idx.push(i - 1);
                dst.push(1.0);
            }
            if i + 1 < n {
                idx.push(i + 1);
                dst.push(1.0);
            }
            indices.push(idx);
            distances.push(dst);
        }
        (indices, distances)
    }

    /// A mutual edge is stored once at its distance rather than summed twice.
    #[test]
    fn test_symmetric_graph_does_not_double_weight_mutual_edges() {
        // Both cells list each other, which is the case coo_to_csr would sum.
        let indices = vec![vec![1], vec![0]];
        let distances = vec![vec![3.0], vec![3.0]];

        let graph = build_symmetric_knn_graph(&indices, &distances).unwrap();

        assert_eq!(graph.get_nnz(), 2);
        for &w in &graph.data {
            assert_relative_eq!(w, 3.0, epsilon = 1e-6);
        }
    }

    /// A one-sided kNN hit still produces an edge in both rows.
    #[test]
    fn test_symmetric_graph_is_symmetric_for_one_sided_edges() {
        // Only cell 0 lists cell 1; the edge must still exist both ways.
        let indices = vec![vec![1], vec![]];
        let distances = vec![vec![2.0], vec![]];

        let graph = build_symmetric_knn_graph(&indices, &distances).unwrap();

        assert_eq!(graph.indptr[1] - graph.indptr[0], 1);
        assert_eq!(graph.indptr[2] - graph.indptr[1], 1);
    }

    /// A split graph comes back as one reachable component with the bridge counted.
    #[test]
    fn test_connect_graph_bridges_two_islands() {
        // Two disjoint pairs, placed so the closest cross pair is (1, 2).
        let indices = vec![vec![1], vec![0], vec![3], vec![2]];
        let distances = vec![vec![1.0], vec![1.0], vec![1.0], vec![1.0]];
        let data = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];

        let graph = build_symmetric_knn_graph(&indices, &distances).unwrap();
        let (repaired, added) = connect_graph(&graph, &data, 0).unwrap();

        assert_eq!(added, 1);
        let (n_comps, _) = connected_components(&repaired).unwrap();
        assert_eq!(n_comps, 1);

        let mut dist = vec![0.0; 4];
        dijkstra_from_source(&repaired, 0, &mut dist).unwrap();
        assert!(dist.iter().all(|d| d.is_finite()));
    }

    /// An already connected graph gains no edges.
    #[test]
    fn test_connect_graph_leaves_connected_input_alone() {
        let (indices, distances) = chain_knn(6);
        let data: Vec<Vec<f32>> = (0..6).map(|i| vec![i as f32]).collect();

        let graph = build_symmetric_knn_graph(&indices, &distances).unwrap();
        let (repaired, added) = connect_graph(&graph, &data, 0).unwrap();

        assert_eq!(added, 0);
        assert_eq!(repaired.get_nnz(), graph.get_nnz());
    }

    /// Every cell's waypoint weights form a distribution.
    #[test]
    fn test_weight_columns_sum_to_one() {
        let (indices, distances) = chain_knn(40);
        let graph = build_symmetric_knn_graph(&indices, &distances).unwrap();
        let waypoints: Vec<usize> = (0..40).step_by(5).collect();

        let res = compute_pseudotime(&graph, &waypoints, 0, 25, Verbosity::Quiet).unwrap();

        for c in 0..res.n_cells {
            let sum: f64 = (0..res.n_waypoints)
                .map(|w| res.weights[w * res.n_cells + c] as f64)
                .sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
        }
    }

    /// Pseudotime rises strictly along a chain and spans the full 0 to 1 range.
    #[test]
    fn test_pseudotime_monotone_on_a_chain() {
        let (indices, distances) = chain_knn(40);
        let graph = build_symmetric_knn_graph(&indices, &distances).unwrap();
        let waypoints: Vec<usize> = std::iter::once(0).chain((5..40).step_by(5)).collect();

        let res = compute_pseudotime(&graph, &waypoints, 0, 25, Verbosity::Quiet).unwrap();

        assert_relative_eq!(res.pseudotime[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(res.pseudotime[39], 1.0, epsilon = 1e-6);
        for i in 0..39 {
            assert!(
                res.pseudotime[i] < res.pseudotime[i + 1],
                "pseudotime not increasing at {i}: {} then {}",
                res.pseudotime[i],
                res.pseudotime[i + 1]
            );
        }
    }

    /// The fused refinement kernel agrees with a literal transcription of the reference.
    #[test]
    fn test_fused_kernel_matches_naive() {
        // Deliberately not a multiple of CELL_BLOCK, so the tail block runs too.
        let (n_wp, n_cells) = (7usize, 23usize);

        let geodesic: Vec<f64> = (0..n_wp * n_cells)
            .map(|i| 1.0 + ((i * 37) % 19) as f64 * 0.5)
            .collect();
        let weights: Vec<f32> = (0..n_wp * n_cells)
            .map(|i| 0.05 + ((i * 11) % 7) as f32 * 0.1)
            .collect();
        let pseudotime: Vec<f64> = (0..n_cells).map(|i| ((i * 5) % 13) as f64 * 0.25).collect();
        let waypoints: Vec<usize> = (0..n_wp).map(|w| (w * 3) % n_cells).collect();

        let mut fused = vec![0.0f64; n_cells];
        refine_once(
            &geodesic,
            &weights,
            &pseudotime,
            &waypoints,
            n_cells,
            &mut fused,
        );

        // Literal transcription of the reference: build the sign matrix against
        // the un-zeroed waypoint times, then override the start row.
        let mut t_wp: Vec<f64> = waypoints.iter().map(|&w| pseudotime[w]).collect();
        let mut signs = vec![1.0f64; n_wp * n_cells];
        for w in 0..n_wp {
            for c in 0..n_cells {
                if pseudotime[c] < t_wp[w] {
                    signs[w * n_cells + c] = -1.0;
                }
            }
        }
        t_wp[0] = 0.0;
        for c in 0..n_cells {
            signs[c] = 1.0;
        }

        let mut naive = vec![0.0f64; n_cells];
        for w in 0..n_wp {
            for c in 0..n_cells {
                let p = geodesic[w * n_cells + c] * signs[w * n_cells + c] + t_wp[w];
                naive[c] += p * weights[w * n_cells + c] as f64;
            }
        }

        for c in 0..n_cells {
            assert_relative_eq!(fused[c], naive[c], epsilon = 1e-12);
        }
    }

    /// A cell whose Gaussian weights all underflow still gets a usable column.
    #[test]
    fn test_weights_survive_an_underflowing_geodesic() {
        // One cell sits hundreds of bandwidths away from every waypoint while
        // the rest are packed together, which is what a sparsely sampled arm of
        // a manifold looks like. The test proves the regime first: it computes
        // the unshifted weights itself and asserts that cell's column sums to
        // exactly zero, which is the division the reference then propagates NaN
        // from. The shift has to recover a usable column from the same input.
        let (n_wp, n_cells) = (4usize, 200usize);
        let far = n_cells - 1;

        let mut geodesic = vec![0.0f64; n_wp * n_cells];
        for wp in 0..n_wp {
            for c in 0..n_cells {
                geodesic[wp * n_cells + c] = if c == far {
                    600.0 + wp as f64
                } else {
                    1.0 + ((wp + c) % 3) as f64 * 0.5
                };
            }
        }

        let sdv = population_standard_deviation(&geodesic)
            * SILVERMAN_FACTOR
            * (n_wp as f64 * n_cells as f64).powf(SILVERMAN_EXPONENT);
        let naive_sum: f64 = (0..n_wp)
            .map(|wp| {
                let d = geodesic[wp * n_cells + far];
                ((-0.5 * (d / sdv) * (d / sdv)).exp() as f32) as f64
            })
            .sum();
        assert_eq!(
            naive_sum, 0.0,
            "the fixture does not underflow, so it cannot cover the shift"
        );

        let weights = gaussian_waypoint_weights(&geodesic, n_wp, n_cells).unwrap();

        assert!(weights.iter().all(|w| w.is_finite()));
        for c in 0..n_cells {
            let sum: f64 = (0..n_wp).map(|w| weights[w * n_cells + c] as f64).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
        }
        // The far cell's nearest waypoint must carry essentially all of it.
        assert!(weights[far] > 0.9, "nearest weight is {}", weights[far]);
    }

    /// Disagreeing directions keep the smaller distance, whatever the scan order.
    #[test]
    fn test_symmetric_graph_keeps_the_minimum_weight_on_a_disagreeing_pair() {
        // Both cells list each other but at different distances, which happens
        // whenever the two directions round differently. The smaller weight has
        // to win, in both input orders, or the CSR depends on scan order.
        let indices = vec![vec![1], vec![0]];
        let forward = vec![vec![2.0], vec![5.0]];
        let backward = vec![vec![5.0], vec![2.0]];

        for distances in [forward, backward] {
            let graph = build_symmetric_knn_graph(&indices, &distances).unwrap();
            assert_eq!(graph.get_nnz(), 2);
            for &w in &graph.data {
                assert_relative_eq!(w, 2.0, epsilon = 1e-6);
            }
        }
    }

    /// Zero cells give an empty graph rather than an error or a panic.
    #[test]
    fn test_symmetric_graph_accepts_empty_input() {
        let graph = build_symmetric_knn_graph(&[], &[]).unwrap();

        assert_eq!(graph.shape, (0, 0));
        assert_eq!(graph.get_nnz(), 0);
    }

    /// Column indices within a CSR row are strictly ascending.
    #[test]
    fn test_csr_rows_come_out_sorted() {
        // The scatter order is what makes rows ascending; a regression there is
        // invisible in every other assertion in this module.
        let (indices, distances) = chain_knn(12);
        let graph = build_symmetric_knn_graph(&indices, &distances).unwrap();

        for i in 0..12 {
            let (lo, hi) = (graph.indptr[i] as usize, graph.indptr[i + 1] as usize);
            assert!(
                graph.indices[lo..hi].windows(2).all(|w| w[0] < w[1]),
                "row {i} is not ascending: {:?}",
                &graph.indices[lo..hi]
            );
        }
    }

    /// Regression: non-finite coordinates priced every bridge at NaN and the repair loop never terminated.
    #[test]
    fn test_connect_graph_gives_up_instead_of_spinning() {
        // Two islands, and the coordinates of the second are non-finite, so
        // every bridging edge is priced at NaN and Dijkstra skips it. Before the
        // cap this looped forever, rebuilding the whole CSR on each pass while
        // `extra` grew without bound.
        let indices = vec![vec![1], vec![0], vec![3], vec![2]];
        let distances = vec![vec![1.0], vec![1.0], vec![1.0], vec![1.0]];
        let data = vec![vec![0.0f32], vec![1.0], vec![f32::NAN], vec![f32::NAN]];

        let graph = build_symmetric_knn_graph(&indices, &distances).unwrap();

        assert!(matches!(
            connect_graph(&graph, &data, 0),
            Err(BixverseErrors::PalantirDisconnectedGraph {
                n_unreachable: 2,
                ..
            })
        ));
    }

    /// Regression: fewer coordinate rows than graph nodes indexed past the end of `data`.
    #[test]
    fn test_connect_graph_rejects_mismatched_data_rows() {
        let indices = vec![vec![1], vec![0], vec![3], vec![2]];
        let distances = vec![vec![1.0], vec![1.0], vec![1.0], vec![1.0]];
        let graph = build_symmetric_knn_graph(&indices, &distances).unwrap();

        // Two rows of coordinates for a four-node graph used to index past the
        // end of `data` while pricing the bridging edge.
        let data = vec![vec![0.0], vec![1.0]];

        assert!(matches!(
            connect_graph(&graph, &data, 0),
            Err(BixverseErrors::DimensionMisMatchSparse { .. })
        ));
    }

    /// A cap that cannot run a single refinement pass is refused up front.
    #[test]
    fn test_pseudotime_rejects_a_useless_iteration_cap() {
        let (indices, distances) = chain_knn(20);
        let graph = build_symmetric_knn_graph(&indices, &distances).unwrap();
        let waypoints: Vec<usize> = (0..20).step_by(4).collect();

        for cap in [0usize, 1usize] {
            assert!(
                matches!(
                    compute_pseudotime(&graph, &waypoints, 0, cap, Verbosity::Quiet),
                    Err(BixverseErrors::PalantirMaxIterationsTooSmall { .. })
                ),
                "a cap of {cap} was silently accepted"
            );
        }

        // Two is the smallest cap that runs a pass.
        let res = compute_pseudotime(&graph, &waypoints, 0, 2, Verbosity::Quiet).unwrap();
        assert_eq!(res.iterations, 1);
    }

    /// The start cell has to be the first waypoint, since the refinement zeroes that row.
    #[test]
    fn test_pseudotime_rejects_a_start_cell_that_is_not_waypoint_zero() {
        let (indices, distances) = chain_knn(20);
        let graph = build_symmetric_knn_graph(&indices, &distances).unwrap();
        let waypoints: Vec<usize> = (0..20).step_by(4).collect();

        assert!(matches!(
            compute_pseudotime(&graph, &waypoints, 8, 25, Verbosity::Quiet),
            Err(BixverseErrors::PalantirStartCellNotFirstWaypoint {
                start_cell: 8,
                first_waypoint: 0
            })
        ));
    }

    /// Unreachable cells are counted per cell, not per geodesic matrix entry.
    #[test]
    fn test_unreachable_count_is_in_cells_not_matrix_entries() {
        // Two waypoints, four cells, cells 2 and 3 unreachable from both. The
        // entry count is four; the cell count is two.
        let n_cells = 4usize;
        let mut geodesic = vec![0.0f64; 2 * n_cells];
        for wp in 0..2 {
            geodesic[wp * n_cells + 2] = f64::INFINITY;
            geodesic[wp * n_cells + 3] = f64::INFINITY;
        }

        assert_eq!(count_unreachable_cells(&geodesic, 2, n_cells), 2);
    }

    /// Cells no waypoint can reach are reported rather than carried into the result.
    #[test]
    fn test_pseudotime_rejects_a_disconnected_graph() {
        let indices = vec![vec![1], vec![0], vec![3], vec![2]];
        let distances = vec![vec![1.0], vec![1.0], vec![1.0], vec![1.0]];
        let graph = build_symmetric_knn_graph(&indices, &distances).unwrap();

        let err = compute_pseudotime(&graph, &[0, 1], 0, 25, Verbosity::Quiet);

        // Not `PalantirDisconnectedGraph`: that variant reports a repair count,
        // which this call site does not have.
        assert!(matches!(
            err,
            Err(BixverseErrors::PalantirUnreachableFromWaypoints { n_unreachable: 2 })
        ));
    }
}
