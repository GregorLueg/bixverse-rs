//! Simulated-annealing partition optimiser.
//!
//! Faithful port of upstream MC2's `OptimizePartitions` C++ class. Given a
//! weighted directed graph, an initial partition assignment, and per-cell UMI
//! counts plus size/UMI bounds, refines the partition by repeatedly moving
//! individual nodes between partitions to maximise a stability score adjusted
//! by a mass/size budget penalty.
//!
//! ### Score model
//!
//! For each partition `p`, each node `i` carries a `NodeScore` summarising
//! the edge weights between `i` and `p`'s members:
//! * `out_weight[p,i]` = sum of outgoing weights from `i` to nodes in `p`.
//! * `in_weight[p,i]` = sum of incoming weights to `i` from nodes in `p`.
//! * `score[p,i] = log2(EPSILON + out_weight * in_weight) / 2`.
//!
//! The total objective is:
//!
//! ```text
//! total = N*log2(N) - incoming_scale
//!       + sum_p [ score_of_partition[p] - size[p]*log2(size[p])
//!                + size[p] * min(mass_factor[p], size_factor[p]) ]
//!       + orphans_count * default_score        (only if with_orphans)
//! ```
//!
//! `mass_factor` is a piecewise log2 budget penalty (zero at `target`,
//! mildly negative inside `[low, high]`, sharply negative outside).
//!
//! Per-move acceptance picks the best candidate partition for the moving
//! node by combining a "hot" diff (just the local score change for the
//! node) and a "cold" diff (the full effect on partition scores including
//! all neighbours). Temperature interpolates between them. Connectivity
//! is preserved as a hard tier — moves that disconnect the source partition
//! are only allowed if they reconnect via the destination.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::core::math::sparse::CompressedSparseData2;

use super::params::PartitionParams;

////////////
// Consts //
////////////

const EPSILON: f64 = 1e-6;
const INNER_SLOPE: f64 = 0.02;
const OUTER_SLOPE: f64 = 0.5;

///////////
// Score //
///////////

/// Per-(partition, node) edge-weight summary and cached log-product score.
///
/// Tracks the total weight of edges flowing out of `node` into a given
/// partition's members, and into `node` from that partition's members. The
/// `score` field is kept in sync via `rescore` and avoids recomputing the log
/// on every candidate evaluation.
#[derive(Clone, Copy, Debug)]
struct NodeScore {
    /// Sum of outgoing edge weights from this node to the partition's members.
    out_weight: f64,
    /// Sum of incoming edge weights to this node from the partition's members.
    in_weight: f64,
    /// Cached `log2(EPSILON + out_weight * in_weight) / 2`.
    score: f64,
}

impl NodeScore {
    /// Construct a `NodeScore` with zero weights and the default epsilon score.
    ///
    /// ### Returns
    ///
    /// A `NodeScore` with `out_weight = 0`, `in_weight = 0`, and
    /// `score = log2(EPSILON) / 2`.
    fn new() -> Self {
        Self {
            out_weight: 0.0,
            in_weight: 0.0,
            score: (EPSILON.log2()) / 2.0,
        }
    }

    /// Adjust the outgoing weight by `direction * w` and clamp to `>= 0`.
    ///
    /// ### Params
    ///
    /// * `direction` - `+1` to add the edge contribution, `-1` to remove it.
    /// * `w` - Edge weight magnitude.
    #[inline]
    fn update_outgoing(&mut self, direction: i32, w: f64) {
        self.out_weight += direction as f64 * w;
        if self.out_weight < 0.0 {
            self.out_weight = 0.0;
        }
    }

    /// Adjust the incoming weight by `direction * w` and clamp to `>= 0`.
    ///
    /// ### Params
    ///
    /// * `direction` - `+1` to add the edge contribution, `-1` to remove it.
    /// * `w` - Edge weight magnitude.
    #[inline]
    fn update_incoming(&mut self, direction: i32, w: f64) {
        self.in_weight += direction as f64 * w;
        if self.in_weight < 0.0 {
            self.in_weight = 0.0;
        }
    }

    /// Count active connection sides: `0`, `1`, or `2`.
    ///
    /// An outgoing side is active when `out_weight > EPSILON`; an incoming side
    /// when `in_weight > 0`. Used as a hard-tier connectivity check before
    /// accepting a move.
    ///
    /// ### Returns
    ///
    /// The number of active sides as an `i32` in `{0, 1, 2}`.
    #[inline]
    fn connectivity(&self) -> i32 {
        let out = if self.out_weight > EPSILON { 1 } else { 0 };
        let inc = if self.in_weight > 0.0 { 1 } else { 0 };
        out + inc
    }

    /// Recompute and cache the log-product score from current weights.
    ///
    /// ### Returns
    ///
    /// The updated score `log2(EPSILON + out_weight * in_weight) / 2`.
    #[inline]
    fn rescore(&mut self) -> f64 {
        self.score = (EPSILON + self.out_weight * self.in_weight).log2() / 2.0;
        self.score
    }
}

///////////////////////////////
// Mass / size budget factor //
///////////////////////////////

/// Piecewise-log2 budget penalty centred on `target`.
///
/// Returns `0` at `target`, mildly negative inside `[low, high]` (slope
/// `INNER_SLOPE`), and sharply negative outside (slope `OUTER_SLOPE`).
///
/// ### Params
///
/// * `mass` - Current partition mass or size.
/// * `low` - Lower bound of the acceptable range.
/// * `target` - Ideal mass or size; penalty is zero here.
/// * `high` - Upper bound of the acceptable range.
///
/// ### Returns
///
/// A non-positive `f64` penalty value.
fn mass_factor(mass: f64, low: f64, target: f64, high: f64) -> f64 {
    if mass < low {
        ((1.0 - INNER_SLOPE) * low / (low + (1.0 / (1.0 - OUTER_SLOPE) - 1.0) * (low - mass)))
            .log2()
    } else if mass < target {
        (1.0 - INNER_SLOPE * (target - mass) / (target - low)).log2()
    } else if mass < high {
        (1.0 - INNER_SLOPE * (mass - target) / (high - target)).log2()
    } else {
        ((1.0 - INNER_SLOPE) * low / (low + (1.0 / (1.0 - OUTER_SLOPE) - 1.0) * (mass - high)))
            .log2()
    }
}

////////////////
// Optimiser  //
////////////////

/// Compute `sum_i log2(total_incoming_weight[i])` over all nodes.
///
/// This constant shift is subtracted from the total score and never changes
/// across moves, so it is computed once at construction.
///
/// ### Params
///
/// * `incoming` - CSR incoming-neighbour graph.
///
/// ### Returns
///
/// The summed log incoming weight as `f64`.
fn initial_incoming_scale(incoming: &CompressedSparseData2<f32, f32>) -> f64 {
    let n = incoming.shape.0;
    let mut acc = 0.0f64;
    for i in 0..n {
        let s: f64 = (incoming.indptr[i]..incoming.indptr[i + 1])
            .map(|idx| incoming.data[idx as usize] as f64)
            .sum();
        if s > 0.0 {
            acc += s.log2();
        }
    }
    acc
}

/// Count the number of nodes assigned to each partition.
///
/// ### Params
///
/// * `partition_of_nodes` - Current assignment vector; entries `< 0` are
///   ignored.
///
/// ### Returns
///
/// A `Vec<usize>` of length `n_partitions` with per-partition node counts.
fn initial_size_of_partitions(partition_of_nodes: &[i32]) -> Vec<usize> {
    let n_partitions = partition_of_nodes
        .iter()
        .copied()
        .max()
        .unwrap_or(-1)
        .max(0) as usize
        + 1;
    let mut sizes = vec![0usize; n_partitions];
    for &p in partition_of_nodes {
        if p >= 0 {
            sizes[p as usize] += 1;
        }
    }
    sizes
}

/// Sum per-node masses into per-partition totals.
///
/// ### Params
///
/// * `partition_of_nodes` - Current assignment vector; entries `< 0` are
///   ignored.
/// * `mass_of_nodes` - Per-node UMI counts.
/// * `n_partitions` - Length of the output vector.
///
/// ### Returns
///
/// A `Vec<f64>` of length `n_partitions` with per-partition mass totals.
fn initial_mass_of_partitions(
    partition_of_nodes: &[i32],
    mass_of_nodes: &[f32],
    n_partitions: usize,
) -> Vec<f64> {
    let mut masses = vec![0.0f64; n_partitions];
    for (i, &p) in partition_of_nodes.iter().enumerate() {
        if p >= 0 {
            masses[p as usize] += mass_of_nodes[i] as f64;
        }
    }
    masses
}

/// Build the dense `[partition][node]` `NodeScore` table from the graph.
///
/// Walks all outgoing edges once and accumulates `out_weight` and `in_weight`
/// contributions for each `(partition, node)` pair according to the current
/// assignment.
///
/// ### Params
///
/// * `outgoing` - CSR outgoing-neighbour graph.
/// * `partition_of_nodes` - Current node-to-partition assignment.
/// * `n_partitions` - Number of partitions.
///
/// ### Returns
///
/// A `Vec<Vec<NodeScore>>` of shape `[n_partitions][n]` with rescored entries.
fn initial_score_table(
    outgoing: &CompressedSparseData2<f32, f32>,
    partition_of_nodes: &[i32],
    n_partitions: usize,
) -> Vec<Vec<NodeScore>> {
    let n = outgoing.shape.0;
    let mut table: Vec<Vec<NodeScore>> = (0..n_partitions)
        .map(|_| vec![NodeScore::new(); n])
        .collect();

    for src in 0..n {
        let src_p = partition_of_nodes[src];
        let start = outgoing.indptr[src] as usize;
        let end = outgoing.indptr[src + 1] as usize;
        for idx in start..end {
            let dst = outgoing.indices[idx] as usize;
            let w = outgoing.data[idx] as f64;
            let dst_p = partition_of_nodes[dst];

            if dst_p >= 0 {
                let s = &mut table[dst_p as usize][src];
                s.update_outgoing(1, w);
                s.rescore();
            }
            if src_p >= 0 {
                let s = &mut table[src_p as usize][dst];
                s.update_incoming(1, w);
                s.rescore();
            }
        }
    }

    table
}

/// Sum each node's score in its assigned partition to get per-partition totals.
///
/// ### Params
///
/// * `partition_of_nodes` - Current assignment vector.
/// * `table` - Dense `[partition][node]` score table.
/// * `n_partitions` - Number of partitions.
///
/// ### Returns
///
/// A `Vec<f64>` of length `n_partitions` with summed node scores.
fn initial_score_of_partitions(
    partition_of_nodes: &[i32],
    table: &[Vec<NodeScore>],
    n_partitions: usize,
) -> Vec<f64> {
    let mut scores = vec![0.0f64; n_partitions];
    for (i, &p) in partition_of_nodes.iter().enumerate() {
        if p >= 0 {
            scores[p as usize] += table[p as usize][i].score;
        }
    }
    scores
}

/// SA partition optimiser. One instance per optimisation run; not reusable.
struct OptimizePartitions<'a> {
    /// CSR outgoing-neighbour graph; row `i` lists cell `i`'s outgoing edges.
    outgoing: &'a CompressedSparseData2<f32, f32>,
    /// CSR incoming-neighbour graph; row `i` lists cell `i`'s incoming edges.
    incoming: &'a CompressedSparseData2<f32, f32>,
    /// Number of nodes.
    n: usize,
    /// Per-node UMI counts used for mass budget penalty.
    mass_of_nodes: &'a [f32],
    /// Lower bound of the acceptable UMI mass range.
    low_mass: f64,
    /// Ideal UMI mass per partition; penalty is zero here.
    target_mass: f64,
    /// Upper bound of the acceptable UMI mass range.
    high_mass: f64,
    /// Lower bound of the acceptable cell-count range.
    low_size: f64,
    /// Ideal cell count per partition; penalty is zero here.
    target_size: f64,
    /// Upper bound of the acceptable cell-count range.
    high_size: f64,
    /// Current node-to-partition assignment; mutated in place during optimisation.
    partition_of_nodes: &'a mut [i32],
    /// Precomputed `sum_i log2(total_incoming[i])`; constant across moves.
    incoming_scale: f64,
    /// Dense `[partition][node]` score table; updated incrementally on each move.
    table: Vec<Vec<NodeScore>>,
    /// Current number of nodes assigned to each partition.
    size_of_partitions: Vec<usize>,
    /// Current total UMI mass assigned to each partition.
    mass_of_partitions: Vec<f64>,
    /// Current sum of node scores within each partition.
    score_of_partitions: Vec<f64>,
    /// Total number of partitions.
    n_partitions: usize,
    /// Per-partition hot mask. `true` means SA may freely move nodes in this
    /// partition; `false` freezes them at `cold_temperature`.
    hot_partitions: Vec<bool>,
    /// Per-node temperature; decremented on rejection, drives the cooling schedule.
    temperature_of_nodes: Vec<f64>,
}

impl<'a> OptimizePartitions<'a> {
    /// Construct a new optimiser and initialise all derived state.
    ///
    /// ### Params
    ///
    /// * `outgoing`, `incoming` - CSR graph and its incoming-neighbour
    ///   transpose.
    /// * `mass_of_nodes` - Per-node UMI counts.
    /// * `low_mass`, `target_mass`, `high_mass` - UMI budget bounds; must
    ///   satisfy `0 < low < target < high`.
    /// * `low_size`, `target_size`, `high_size` - Cell-count budget bounds;
    ///   same constraint.
    /// * `partition_of_nodes` - Initial assignment; borrowed mutably for the
    ///   lifetime of `self`.
    /// * `hot_partitions` - Per-partition hot mask; padded to `n_partitions`
    ///   with `false` if short.
    ///
    /// ### Panics
    ///
    /// Panics if dimension invariants or bound ordering are violated.
    #[allow(clippy::too_many_arguments)]
    fn new(
        outgoing: &'a CompressedSparseData2<f32, f32>,
        incoming: &'a CompressedSparseData2<f32, f32>,
        mass_of_nodes: &'a [f32],
        low_mass: f64,
        target_mass: f64,
        high_mass: f64,
        low_size: f64,
        target_size: f64,
        high_size: f64,
        partition_of_nodes: &'a mut [i32],
        hot_partitions: Vec<bool>,
    ) -> Self {
        let n = outgoing.shape.0;
        assert_eq!(incoming.shape.0, n);
        assert_eq!(partition_of_nodes.len(), n);
        assert_eq!(mass_of_nodes.len(), n);
        assert!(0.0 < low_mass && low_mass < target_mass && target_mass < high_mass);
        assert!(0.0 < low_size && low_size < target_size && target_size < high_size);

        let size_of_partitions = initial_size_of_partitions(partition_of_nodes);
        let n_partitions = size_of_partitions.len();
        let incoming_scale = initial_incoming_scale(incoming);
        let table = initial_score_table(outgoing, partition_of_nodes, n_partitions);
        let mass_of_partitions =
            initial_mass_of_partitions(partition_of_nodes, mass_of_nodes, n_partitions);
        let score_of_partitions =
            initial_score_of_partitions(partition_of_nodes, &table, n_partitions);

        // Pad hot mask to n_partitions if needed.
        let mut hot = hot_partitions;
        if hot.len() < n_partitions {
            hot.resize(n_partitions, false);
        }

        Self {
            outgoing,
            incoming,
            n,
            mass_of_nodes,
            low_mass,
            target_mass,
            high_mass,
            low_size,
            target_size,
            high_size,
            partition_of_nodes,
            incoming_scale,
            table,
            size_of_partitions,
            mass_of_partitions,
            score_of_partitions,
            n_partitions,
            hot_partitions: hot,
            temperature_of_nodes: vec![1.0; n],
        }
    }

    /// Compute the global objective score from current state.
    ///
    /// ### Params
    ///
    /// * `with_orphans` - If `true`, includes a default-score term for nodes
    ///   assigned to partition `-1` so scores are comparable across runs with
    ///   differing numbers of assigned cells.
    ///
    /// ### Returns
    ///
    /// The per-node normalised score as `f64`.
    fn score(&self, with_orphans: bool) -> f64 {
        let mut total = (self.n as f64) * (self.n as f64).log2() - self.incoming_scale;
        let mut orphans = self.n;
        for p in 0..self.n_partitions {
            let s = self.score_of_partitions[p];
            let m = self.mass_of_partitions[p];
            let sz = self.size_of_partitions[p];
            if sz > 0 {
                total += s - (sz as f64) * (sz as f64).log2();
                let mf = mass_factor(m, self.low_mass, self.target_mass, self.high_mass);
                let sf = mass_factor(sz as f64, self.low_size, self.target_size, self.high_size);
                total += (sz as f64) * mf.min(sf);
            }
            orphans -= sz;
        }
        if with_orphans {
            total += (orphans as f64) * NodeScore::new().score;
            total / self.n as f64
        } else {
            total / (self.n - orphans) as f64
        }
    }

    /// Drive the SA loop to convergence.
    ///
    /// Shuffles nodes each pass and attempts one improvement per node. The
    /// per-node temperature is cooled on rejection by `cooldown_node`; the
    /// global pass temperature decays at rate `1 - cooldown_pass / n`.
    /// Terminates when a zero-temperature pass produces no improvement.
    ///
    /// ### Params
    ///
    /// * `random_seed` - RNG seed for node shuffling and weighted picks.
    /// * `cooldown_pass` - Fractional temperature decay per node visit; in
    ///   `(0, 1)`.
    /// * `cooldown_node` - Per-node temperature multiplier on rejection; in
    ///   `[0, 1]`.
    /// * `cold_temperature` - Starting temperature for nodes in non-hot
    ///   partitions.
    fn optimise(
        &mut self,
        random_seed: u64,
        cooldown_pass: f64,
        cooldown_node: f64,
        cold_temperature: f64,
    ) {
        assert!((0.0..=1.0).contains(&cooldown_node));
        if cooldown_pass != 0.0 {
            assert!(cooldown_pass > 0.0 && cooldown_pass < 1.0);
        }

        let mut rng = StdRng::seed_from_u64(random_seed);

        let mut frozen = vec![false; self.n];
        for i in 0..self.n {
            let p = self.partition_of_nodes[i];
            if (0..self.n_partitions as i32).contains(&p) && !self.hot_partitions[p as usize] {
                self.temperature_of_nodes[i] = cold_temperature;
                frozen[i] = true;
            } else if p < -1 {
                frozen[i] = true;
            } else {
                self.temperature_of_nodes[i] = 1.0;
            }
        }

        let cooldown_rate = 1.0 - cooldown_pass / self.n as f64;
        let mut temperature = 1.0_f64;

        let mut indices: Vec<usize> = (0..self.n).collect();
        let mut did_improve = true;
        let mut did_skip = false;

        while temperature > 0.0 || did_improve {
            if did_improve {
                did_improve = false;
                did_skip = false;
            } else if did_skip {
                did_skip = false;
                let mut next_t = temperature;
                for i in 0..self.n {
                    if !frozen[i] && self.temperature_of_nodes[i] < next_t {
                        next_t = self.temperature_of_nodes[i];
                    }
                }
                temperature = next_t;
            } else {
                temperature = 0.0;
            }

            indices.shuffle(&mut rng);
            for &i in &indices {
                temperature *= cooldown_rate;
                if self.temperature_of_nodes[i] < temperature {
                    did_skip = true;
                    continue;
                }
                if self.improve_node(i, &mut rng, temperature) {
                    frozen[i] = false;
                    did_improve = true;
                } else {
                    frozen[i] = false;
                    self.temperature_of_nodes[i] = temperature * (1.0 - cooldown_node);
                }
            }
        }
    }

    /// Attempt to move `node_index` to a better partition.
    ///
    /// Computes hot and cold diffs across all candidate partitions, classifies
    /// them into tiers, and applies the best move found.
    ///
    /// ### Params
    ///
    /// * `node_index` - Index of the node to consider moving.
    /// * `rng` - RNG used for weighted candidate selection.
    /// * `temperature` - Current SA temperature; interpolates hot/cold diffs.
    ///
    /// ### Returns
    ///
    /// `true` if a move was applied, `false` otherwise.
    fn improve_node(&mut self, node_index: usize, rng: &mut StdRng, temperature: f64) -> bool {
        let current_p = self.partition_of_nodes[node_index];
        if current_p < 0 || self.size_of_partitions[current_p as usize] < 2 {
            return false;
        }
        let current_p = current_p as usize;

        // tmp buffers — could be cached on self for hot loops, but the
        // allocations here are tiny relative to the per-node work.
        let mut cold_diffs = vec![0.0f64; self.n_partitions];
        let mut hot_diffs = vec![0.0f64; self.n_partitions];
        let mut conn_diffs = vec![0.0f64; self.n_partitions];
        let mut disc_diffs = vec![0.0f64; self.n_partitions];

        self.collect_initial_diffs(node_index, current_p, &mut cold_diffs, &mut hot_diffs);
        let current_conn = self.table[current_p][node_index].connectivity();
        self.collect_cold_diffs(
            node_index,
            current_p,
            current_conn,
            &mut cold_diffs,
            &mut conn_diffs,
            &mut disc_diffs,
        );

        // Build candidate partition lists (both / connectivity / goal), and at
        // the same time compute the total-diff value used for the weighted
        // random pick.
        let (current_cold_diff, total_diffs, candidates) = self.collect_candidates(
            node_index,
            current_p,
            current_conn,
            &cold_diffs,
            &hot_diffs,
            &conn_diffs,
            &disc_diffs,
            temperature,
        );

        let chosen = self.choose_target(&candidates, &total_diffs, rng);
        let Some(chosen_p) = chosen else {
            return false;
        };

        let chosen_cold_diff = cold_diffs[chosen_p];
        self.apply_move(
            node_index,
            current_p,
            chosen_p,
            current_cold_diff,
            chosen_cold_diff,
        );
        true
    }

    /// Compute the initial hot and cold diffs for moving `node_index` out of
    /// `current_p`.
    ///
    /// Both are set to the signed node score contribution: negative for
    /// `current_p` (node leaves), positive for all other partitions (node would
    /// join). Cold diffs are later augmented with neighbour effects in
    /// `collect_cold_diffs`.
    ///
    /// ### Params
    ///
    /// * `node_index` - Node being considered for a move.
    /// * `current_p` - The node's current partition.
    /// * `cold` - Output cold-diff buffer; overwritten in place.
    /// * `hot` - Output hot-diff buffer; overwritten in place.
    fn collect_initial_diffs(
        &self,
        node_index: usize,
        current_p: usize,
        cold: &mut [f64],
        hot: &mut [f64],
    ) {
        for p in 0..self.n_partitions {
            let direction = if p == current_p { -1.0 } else { 1.0 };
            let s = direction * self.table[p][node_index].score;
            cold[p] = s;
            hot[p] = s;
        }
    }

    /// Augment cold diffs with the propagated effect on each neighbour's score.
    ///
    /// For every neighbour `j` of `node_index`, simulates the score change in
    /// `j`'s partition if the move were applied, and accumulates the delta.
    /// Also updates per-partition connectivity and disconnect-flip counters.
    ///
    /// ### Params
    ///
    /// * `node_index` - Node being considered for a move.
    /// * `current_p` - The node's current partition.
    /// * `current_conn` - Connectivity of `node_index` in `current_p`.
    /// * `cold` - Cold-diff buffer; augmented in place.
    /// * `conn` - Per-partition connectivity delta buffer; overwritten in
    ///   place.
    /// * `disc` - Per-partition disconnect-flip buffer; overwritten in place.
    fn collect_cold_diffs(
        &self,
        node_index: usize,
        current_p: usize,
        current_conn: i32,
        cold: &mut [f64],
        conn: &mut [f64],
        disc: &mut [f64],
    ) {
        for p in 0..self.n_partitions {
            if p == current_p {
                conn[p] = 0.0;
                disc[p] = 0.0;
                continue;
            }
            let other_conn = self.table[p][node_index].connectivity();
            conn[p] = (other_conn - current_conn) as f64;
            disc[p] = if current_conn > 0 && other_conn == 0 {
                1.0
            } else if current_conn == 0 && other_conn > 0 {
                -1.0
            } else {
                0.0
            };
        }

        // walk neighbours via merged outgoing/incoming traversal.
        let out_start = self.outgoing.indptr[node_index] as usize;
        let out_end = self.outgoing.indptr[node_index + 1] as usize;
        let in_start = self.incoming.indptr[node_index] as usize;
        let in_end = self.incoming.indptr[node_index + 1] as usize;

        let mut op = out_start;
        let mut ip = in_start;

        while op < out_end || ip < in_end {
            let out_node = if op < out_end {
                self.outgoing.indices[op] as i64
            } else {
                self.n as i64
            };
            let in_node = if ip < in_end {
                self.incoming.indices[ip] as i64
            } else {
                self.n as i64
            };
            let other = out_node.min(in_node) as usize;
            let is_out = out_node == other as i64;
            let is_in = in_node == other as i64;
            let out_w = if is_out {
                self.outgoing.data[op] as f64
            } else {
                0.0
            };
            let in_w = if is_in {
                self.incoming.data[ip] as f64
            } else {
                0.0
            };

            let other_p = self.partition_of_nodes[other];
            if other_p >= 0 {
                let other_p = other_p as usize;
                // direction: +1 if the move pushes a new edge into this
                // partition, -1 if it removes one. The moving node leaves
                // current_p, so weights tied to current_p get -1; weights to
                // all other partitions get +1.
                let direction: i32 = if other_p == current_p { -1 } else { 1 };

                // Snapshot the existing score, simulate the update,
                // measure the delta, and unwind.
                let mut s = self.table[other_p][other];
                let old_score = s.score;
                let old_conn = s.connectivity();
                if is_out {
                    // src -> dst means dst.in_weight changes by ±out_w
                    // wrt the partition that src moves between.
                    s.update_incoming(direction * (is_out as i32), out_w);
                }
                if is_in {
                    s.update_outgoing(direction * (is_in as i32), in_w);
                }
                let new_score = s.rescore();
                let new_conn = s.connectivity();

                cold[other_p] += new_score - old_score;
                conn[other_p] += (new_conn - old_conn) as f64;
                if old_conn == 0 && new_conn > 0 {
                    disc[other_p] -= 1.0;
                }
                if old_conn > 0 && new_conn == 0 {
                    disc[other_p] += 1.0;
                }
            }

            if is_out {
                op += 1;
            }
            if is_in {
                ip += 1;
            }
        }
    }

    /// Build the per-partition candidate lists and total-diff vector.
    ///
    /// Combines hot and cold diffs with the mass/size budget penalty to
    /// produce a `total_diff` per candidate partition. Candidates are
    /// classified into three tiers based on connectivity and score improvement.
    /// Moves that increase total disconnection or fail the connectivity floor
    /// are excluded.
    ///
    /// ### Params
    ///
    /// * `node_index` - Node being considered for a move.
    /// * `current_p` - The node's current partition.
    /// * `current_conn` - Connectivity of `node_index` in `current_p`.
    /// * `cold` - Cold-diff per partition.
    /// * `hot` - Hot-diff per partition.
    /// * `conn` - Connectivity delta per partition.
    /// * `disc` - Disconnect-flip delta per partition.
    /// * `temperature` - Current SA temperature.
    ///
    /// ### Returns
    ///
    /// A tuple of `(current_cold_diff, total_diffs, candidates)`.
    #[allow(clippy::too_many_arguments)]
    fn collect_candidates(
        &self,
        node_index: usize,
        current_p: usize,
        current_conn: i32,
        cold: &[f64],
        hot: &[f64],
        conn: &[f64],
        disc: &[f64],
        temperature: f64,
    ) -> (f64, Vec<f64>, Candidates) {
        let node_mass = self.mass_of_nodes[node_index] as f64;

        let current_hot_diff = hot[current_p];
        let current_cold_diff = cold[current_p];

        let min_conn_diff = if current_conn == 0 { 1 } else { 0 };

        // adjusted (= pure-score plus mass/size term) diff at the source.
        let old_size = self.size_of_partitions[current_p];
        let old_mass = self.mass_of_partitions[current_p];
        let old_mf = mass_factor(old_mass, self.low_mass, self.target_mass, self.high_mass);
        let old_sf = mass_factor(
            old_size as f64,
            self.low_size,
            self.target_size,
            self.high_size,
        );
        let old_score = self.score_of_partitions[current_p];
        let old_adj = old_score - (old_size as f64) * (old_size as f64).log2()
            + (old_size as f64) * old_mf.min(old_sf);

        let new_size = old_size - 1;
        let new_mass = old_mass - node_mass;
        let new_mf = mass_factor(new_mass, self.low_mass, self.target_mass, self.high_mass);
        let new_sf = mass_factor(
            new_size as f64,
            self.low_size,
            self.target_size,
            self.high_size,
        );
        let new_score = old_score + current_cold_diff;
        let new_adj = if new_size > 0 {
            new_score - (new_size as f64) * (new_size as f64).log2()
                + (new_size as f64) * new_mf.min(new_sf)
        } else {
            0.0
        };
        let current_adj_cold_diff = new_adj - old_adj;
        let current_conn_diff = conn[current_p];
        let current_disc_diff = disc[current_p];

        let mut total_diffs = vec![0.0f64; self.n_partitions];
        let mut cands = Candidates::default();

        for p in 0..self.n_partitions {
            if p == current_p {
                total_diffs[p] = 0.0;
                continue;
            }
            let old_size = self.size_of_partitions[p];
            let old_mass = self.mass_of_partitions[p];
            let old_mf = mass_factor(old_mass, self.low_mass, self.target_mass, self.high_mass);
            let old_sf = mass_factor(
                old_size as f64,
                self.low_size,
                self.target_size,
                self.high_size,
            );
            let old_score = self.score_of_partitions[p];
            let old_adj = if old_size > 0 {
                old_score - (old_size as f64) * (old_size as f64).log2()
                    + (old_size as f64) * old_mf.min(old_sf)
            } else {
                0.0
            };

            let new_size = old_size + 1;
            let new_mass = old_mass + node_mass;
            let new_mf = mass_factor(new_mass, self.low_mass, self.target_mass, self.high_mass);
            let new_sf = mass_factor(
                new_size as f64,
                self.low_size,
                self.target_size,
                self.high_size,
            );
            let new_score = old_score + cold[p];
            let new_adj = new_score - (new_size as f64) * (new_size as f64).log2()
                + (new_size as f64) * new_mf.min(new_sf);

            let p_conn_diff = conn[p];
            let adjusted_conn_diff = p_conn_diff + current_conn_diff;
            let p_disc_diff = disc[p];

            // forbid moves that increase total disconnection.
            if p_disc_diff + current_disc_diff > 0.0 {
                continue;
            }
            // forbid moves that fail the connectivity floor.
            if (adjusted_conn_diff as i32) < min_conn_diff {
                continue;
            }

            let adj_cold_diff = new_adj - old_adj;
            let total = (current_hot_diff + hot[p]) * temperature
                + (current_adj_cold_diff + adj_cold_diff) * (1.0 - temperature);
            total_diffs[p] = total;

            if total >= EPSILON {
                if min_conn_diff > 0 {
                    cands.both.push(p);
                } else {
                    cands.goal.push(p);
                }
            } else if min_conn_diff > 0 {
                cands.connectivity.push(p);
            }
        }

        (current_cold_diff, total_diffs, cands)
    }

    /// Pick a target partition from the tiered candidate lists.
    ///
    /// Tries `both` first (improves connectivity and score), then
    /// `connectivity`, then `goal` (improves score only). Within a tier samples
    /// proportionally to `total_diff - min_total_diff`; falls back to a uniform
    /// pick when all weights are equal.
    ///
    /// ### Params
    ///
    /// * `candidates` - Tiered candidate partition lists.
    /// * `total_diffs` - Per-partition total diff values.
    /// * `rng` - RNG used for weighted sampling.
    ///
    /// ### Returns
    ///
    /// The chosen partition index, or `None` if all lists are empty.
    fn choose_target(
        &self,
        candidates: &Candidates,
        total_diffs: &[f64],
        rng: &mut StdRng,
    ) -> Option<usize> {
        let mut pick = |list: &[usize]| -> Option<usize> {
            if list.is_empty() {
                return None;
            }
            let min_total = list
                .iter()
                .map(|&p| total_diffs[p])
                .fold(f64::INFINITY, f64::min);
            let sum: f64 = list.iter().map(|&p| total_diffs[p] - min_total).sum();
            if sum <= 0.0 {
                // all ties at min: pick uniformly.
                return Some(list[rng.random_range(0..list.len())]);
            }
            let mut roll: f64 = rng.random::<f64>() * sum;
            for &p in list {
                roll -= total_diffs[p] - min_total;
                if roll <= EPSILON {
                    return Some(p);
                }
            }
            Some(list[list.len() - 1]) // fallthrough on rounding
        };

        pick(&candidates.both)
            .or_else(|| pick(&candidates.connectivity))
            .or_else(|| pick(&candidates.goal))
    }

    /// Apply a move: update the score table, assignment, sizes, masses, and
    /// partition scores.
    ///
    /// Walks all neighbours of `node_index` once to propagate weight changes
    /// in `from_p` and `to_p`, then updates bookkeeping scalars.
    ///
    /// ### Params
    ///
    /// * `node_index` - Node being moved.
    /// * `from_p` - Source partition.
    /// * `to_p` - Destination partition.
    /// * `from_cold_diff` - Pre-computed score delta for `from_p`.
    /// * `to_cold_diff` - Pre-computed score delta for `to_p`.
    fn apply_move(
        &mut self,
        node_index: usize,
        from_p: usize,
        to_p: usize,
        from_cold_diff: f64,
        to_cold_diff: f64,
    ) {
        // Update neighbour scores in from_p / to_p.
        let out_start = self.outgoing.indptr[node_index] as usize;
        let out_end = self.outgoing.indptr[node_index + 1] as usize;
        let in_start = self.incoming.indptr[node_index] as usize;
        let in_end = self.incoming.indptr[node_index + 1] as usize;

        let mut op = out_start;
        let mut ip = in_start;
        while op < out_end || ip < in_end {
            let out_node = if op < out_end {
                self.outgoing.indices[op] as i64
            } else {
                self.n as i64
            };
            let in_node = if ip < in_end {
                self.incoming.indices[ip] as i64
            } else {
                self.n as i64
            };
            let other = out_node.min(in_node) as usize;
            let is_out = out_node == other as i64;
            let is_in = in_node == other as i64;
            let out_w = if is_out {
                self.outgoing.data[op] as f64
            } else {
                0.0
            };
            let in_w = if is_in {
                self.incoming.data[ip] as f64
            } else {
                0.0
            };

            // Symmetric: in from_p the moving node leaves, so neighbour
            // weights tied to from_p decrease; in to_p they increase.
            let s_from = &mut self.table[from_p][other];
            if is_out {
                s_from.update_incoming(-1, out_w);
            }
            if is_in {
                s_from.update_outgoing(-1, in_w);
            }
            s_from.rescore();

            let s_to = &mut self.table[to_p][other];
            if is_out {
                s_to.update_incoming(1, out_w);
            }
            if is_in {
                s_to.update_outgoing(1, in_w);
            }
            s_to.rescore();

            if is_out {
                op += 1;
            }
            if is_in {
                ip += 1;
            }
        }

        // Update assignment, sizes, masses, partition scores.
        debug_assert_eq!(self.partition_of_nodes[node_index], from_p as i32);
        self.partition_of_nodes[node_index] = to_p as i32;
        self.size_of_partitions[from_p] -= 1;
        self.size_of_partitions[to_p] += 1;
        let m = self.mass_of_nodes[node_index] as f64;
        self.mass_of_partitions[from_p] -= m;
        self.mass_of_partitions[to_p] += m;
        self.score_of_partitions[from_p] += from_cold_diff;
        self.score_of_partitions[to_p] += to_cold_diff;
    }
}

/// Three-tier candidate partition lists used by `choose_target`.
#[derive(Default)]
struct Candidates {
    /// Partitions that improve both connectivity and the score objective.
    both: Vec<usize>,
    /// Partitions that improve connectivity only.
    connectivity: Vec<usize>,
    /// Partitions that improve the score objective only.
    goal: Vec<usize>,
}

//////////
// Main //
//////////

/// Run SA partition optimisation on `partition_of_nodes` in place.
///
/// ### Params
///
/// * `outgoing`, `incoming` - The graph as CSR (outgoing) and the same
///   matrix transposed and re-flagged as CSR for incoming-neighbour
///   access. See `seeds.rs::choose_seeds` for the same convention.
/// * `mass_of_nodes` - Per-node UMI counts (or any positive mass metric).
/// * `partition_of_nodes` - Initial assignment, mutated in place. All
///   entries must be in `[0, n_partitions)`.
/// * `low_mass`, `target_mass`, `high_mass` - UMI budget bounds. Must
///   satisfy `0 < low < target < high`.
/// * `low_size`, `target_size`, `high_size` - Cell-count budget bounds.
/// * `params` - SA temperature parameters.
/// * `cold_temperature` - Starting temperature for nodes in non-hot
///   partitions. Use `cooldown_pass` from params on first call;
///   re-optimisation passes lower this further to avoid disturbing
///   already-converged regions.
/// * `hot_partitions` - Per-partition `bool` mask. `true` means SA may
///   freely move nodes touching this partition; `false` freezes them at
///   `cold_temperature`. Pass `vec![true; n_partitions]` for full
///   optimisation.
/// * `random_seed` - RNG seed.
///
/// ### Returns
///
/// The score after optimisation (`with_orphans = true`).
#[allow(clippy::too_many_arguments)]
pub fn optimise_partitions(
    outgoing: &CompressedSparseData2<f32, f32>,
    incoming: &CompressedSparseData2<f32, f32>,
    mass_of_nodes: &[f32],
    partition_of_nodes: &mut [i32],
    low_mass: f64,
    target_mass: f64,
    high_mass: f64,
    low_size: f64,
    target_size: f64,
    high_size: f64,
    params: &PartitionParams,
    cold_temperature: f64,
    hot_partitions: Vec<bool>,
    random_seed: u64,
) -> f64 {
    let mut opt = OptimizePartitions::new(
        outgoing,
        incoming,
        mass_of_nodes,
        low_mass,
        target_mass,
        high_mass,
        low_size,
        target_size,
        high_size,
        partition_of_nodes,
        hot_partitions,
    );
    opt.optimise(
        random_seed,
        params.cooldown_pass,
        params.cooldown_node,
        cold_temperature,
    );
    opt.score(true)
}

/// Compute the score from a current assignment without modifying it.
///
/// Useful for diagnostics and for the outer orchestration loop's
/// "did this iteration improve?" check.
#[allow(clippy::too_many_arguments)]
pub fn score_partitions(
    outgoing: &CompressedSparseData2<f32, f32>,
    incoming: &CompressedSparseData2<f32, f32>,
    mass_of_nodes: &[f32],
    partition_of_nodes: &mut [i32],
    low_mass: f64,
    target_mass: f64,
    high_mass: f64,
    low_size: f64,
    target_size: f64,
    high_size: f64,
    with_orphans: bool,
) -> f64 {
    let n_partitions = (partition_of_nodes
        .iter()
        .copied()
        .max()
        .unwrap_or(-1)
        .max(0)
        + 1) as usize;
    let opt = OptimizePartitions::new(
        outgoing,
        incoming,
        mass_of_nodes,
        low_mass,
        target_mass,
        high_mass,
        low_size,
        target_size,
        high_size,
        partition_of_nodes,
        vec![false; n_partitions],
    );
    opt.score(with_orphans)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::math::sparse::CompressedSparseFormat, prelude::VecIndexCast};

    /// Builds a simple weighted graph from edge list. Returns
    /// (outgoing CSR, incoming CSR) for a graph of `n` nodes.
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

        // Build incoming = transpose of outgoing, re-flagged as CSR.
        let inc = crate::core::math::sparse::transpose_sparse(&outgoing);
        let incoming = CompressedSparseData2 {
            data: inc.data,
            indices: inc.indices,
            indptr: inc.indptr,
            cs_type: CompressedSparseFormat::Csr,
            data_2: inc.data_2,
            shape: inc.shape,
        };
        (outgoing, incoming)
    }

    /// The mass penalty is zero on target and falls off faster outside the bounds than inside.
    #[test]
    fn mass_factor_is_negative_off_target() {
        let at_target = mass_factor(100.0, 50.0, 100.0, 200.0);
        assert!(
            at_target.abs() < 1e-9,
            "expected 0 at target, got {at_target}"
        );
        let inside_lo = mass_factor(75.0, 50.0, 100.0, 200.0);
        let inside_hi = mass_factor(150.0, 50.0, 100.0, 200.0);
        let outside_lo = mass_factor(20.0, 50.0, 100.0, 200.0);
        let outside_hi = mass_factor(400.0, 50.0, 100.0, 200.0);
        assert!(inside_lo < 0.0 && inside_lo > -0.1);
        assert!(inside_hi < 0.0 && inside_hi > -0.1);
        assert!(outside_lo < inside_lo, "outside-low must be sharper");
        assert!(outside_hi < inside_hi, "outside-high must be sharper");
    }

    /// Optimisation recovers the two cliques from a deliberately wrong start.
    #[test]
    fn two_clique_graph_separates_correctly() {
        // 6 nodes, two triangles {0,1,2} and {3,4,5}, weakly linked
        // 0 <-> 3 only.
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
        let (out, inc) = make_graph(&edges, 6);
        let mass = vec![1.0f32; 6];

        // Start with a deliberately wrong assignment: split each clique.
        let mut assignment = vec![0i32, 1, 0, 1, 0, 1];
        let params = PartitionParams {
            cooldown_pass: 0.05,
            cooldown_node: 0.25,
            ..PartitionParams::default()
        };

        let _score = optimise_partitions(
            &out,
            &inc,
            &mass,
            &mut assignment,
            0.5,
            3.0,
            10.0,
            0.5,
            3.0,
            10.0,
            &params,
            0.05,
            vec![true, true],
            42,
        );

        // All members of the same clique should share a partition.
        assert_eq!(assignment[0], assignment[1], "0 and 1 in same clique");
        assert_eq!(assignment[1], assignment[2], "1 and 2 in same clique");
        assert_eq!(assignment[3], assignment[4], "3 and 4 in same clique");
        assert_eq!(assignment[4], assignment[5], "4 and 5 in same clique");
        assert_ne!(
            assignment[0], assignment[3],
            "cliques must end up separated"
        );
    }

    /// The partition score depends on the grouping, not on the label each group carries.
    #[test]
    fn score_is_invariant_to_partition_relabelling() {
        // Single edge graph; each node in its own partition. The score sums
        // over partitions, so swapping the two labels must not move it.
        let edges = vec![(0, 1, 1.0), (1, 0, 1.0)];
        let (out, inc) = make_graph(&edges, 2);
        let mass = vec![1.0f32; 2];

        let score_of = |assignment: &mut [i32]| {
            score_partitions(
                &out, &inc, &mass, assignment, 0.5, 1.0, 2.0, 0.5, 1.0, 2.0, true,
            )
        };

        let s = score_of(&mut [0i32, 1]);
        let s_swapped = score_of(&mut [1i32, 0]);

        assert!(s.is_finite(), "score must be finite, got {s}");
        assert!(
            (s - s_swapped).abs() < 1e-12,
            "relabelling changed the score: {s} vs {s_swapped}"
        );
    }
}
