//! Leiden and Louvain community detection on modularity.
//!
//! One engine for both: Louvain is Leiden with the refinement phase switched
//! off. Phase semantics follow igraph's `igraph_community_leiden` (fast local
//! moving with a queue of unstable nodes, randomised constrained merging in the
//! refinement, aggregation on the refined partition seeded with the unrefined
//! one). Weights are read in `T`, but every node, cluster and edge total is
//! held in `f64`, since the cluster totals are updated millions of times.
//!
//! Departure from igraph: the graph is relabelled into BFS order first, and
//! Leiden's local moving visits nodes in that order from a seeded random
//! start rather than in a full shuffle. On kNN graphs of unordered cells this
//! keeps the neighbour scans cache-local and needs about half the node
//! evaluations, at the same modularity. Louvain keeps the full shuffle, as the
//! ordered sweep costs it modularity without a refinement to repair it.
//!
//! Local moving defaults to the sequential queue. [`LocalMoving::Coloured`]
//! parallelises it on large levels, see its docs; it stays independent of the
//! thread count.
//!
//! Determinism: the queue is sequential and seeded. Refinement
//! runs in parallel over clusters, each cluster with its own RNG derived from
//! `(seed, iteration, level, cluster)`, and aggregation is parallel over
//! refined clusters with a fixed reduction order. The partition therefore
//! depends on the seed only, not on the thread count.

use num_traits::{ToPrimitive, Zero};
use rand::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use crate::graph::community_detections::remap_communities_by_size;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Nodes in a level below which refinement and aggregation run sequentially.
/// Louvain is also called from inside parallel loops over seeds and
/// resolutions (fast_clusters), where small levels are not worth a fork.
const LEIDEN_PAR_MIN_NODES: usize = 4096;

/// Safety cap on rounds of the parallel local-moving modes. Each round only
/// revisits nodes whose neighbourhood changed, so real graphs stop far below
/// it.
const PAR_MOVE_MAX_ROUNDS: usize = 1000;

/// Nodes per rayon task when deciding moves in parallel. Per-node work follows
/// the degree, so small tasks balance better.
const PAR_MOVE_MIN_LEN: usize = 64;

/// Target id of a move into an empty cluster, resolved when it is applied.
const NEW_CLUSTER: u32 = u32::MAX;

////////////
// Params //
////////////

/// How the local-moving phase visits the nodes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum LocalMoving {
    /// Traag's fast local moving: a sequential queue of unstable nodes
    #[default]
    Queue,
    /// Greedy colour classes of a seeded node order. Each class decides its
    /// moves in parallel against the state frozen at its start (no two nodes
    /// of a class are neighbours, so their edge weights to clusters are
    /// exact), then applies them in node order, re-checking each gain against
    /// the live cluster totals. The scheme is the one in clust-libreville-rs
    /// and leiden-rs.
    Coloured,
}

/// Parse a [`LocalMoving`] name.
///
/// ### Params
///
/// * `s` - `"queue"` or `"coloured"`, case-insensitive
///
/// ### Returns
///
/// The [`LocalMoving`], or `None` for an unknown name
pub fn parse_local_moving(s: &str) -> Option<LocalMoving> {
    match s.to_lowercase().as_str() {
        "queue" => Some(LocalMoving::Queue),
        "coloured" | "colored" => Some(LocalMoving::Coloured),
        _ => None,
    }
}

/// Parameters for [`leiden_sparse_graph`].
///
/// ### Fields
///
/// * `resolution` - Modularity resolution parameter (gamma)
/// * `beta` - Randomness of the refinement merges; the target is drawn with
///   probability proportional to `exp(gain / beta)`. igraph default 0.01.
/// * `n_iterations` - Number of full Leiden iterations, each starting from
///   the previous partition. igraph's R default is 2.
/// * `refine` - Run the Leiden refinement phase. `false` gives Louvain.
/// * `seed` - Seed for the node orders and refinement draws
/// * `local_moving` - How local moving visits the nodes; the parallel modes
///   only run on levels of at least `LEIDEN_PAR_MIN_NODES` (4096) nodes,
///   smaller ones use the queue
#[derive(Clone, Copy, Debug)]
pub struct LeidenParams {
    /// Modularity resolution parameter (gamma)
    pub resolution: f64,
    /// Randomness of the refinement merges
    pub beta: f64,
    /// Number of full iterations
    pub n_iterations: usize,
    /// Run the refinement phase (Leiden) or not (Louvain)
    pub refine: bool,
    /// Random seed
    pub seed: u64,
    /// Local-moving strategy
    pub local_moving: LocalMoving,
}

impl LeidenParams {
    /// Generate new Leiden parameters.
    ///
    /// ### Params
    ///
    /// * `resolution` - Modularity resolution parameter (gamma)
    /// * `beta` - Randomness of the refinement merges
    /// * `n_iterations` - Number of full iterations
    /// * `refine` - Run the refinement phase (Leiden) or not (Louvain)
    /// * `seed` - Random seed
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn new(resolution: f64, beta: f64, n_iterations: usize, refine: bool, seed: u64) -> Self {
        Self {
            resolution,
            beta,
            n_iterations,
            refine,
            seed,
            local_moving: LocalMoving::Queue,
        }
    }
}

/// Defaults match `igraph::cluster_leiden(objective_function = "modularity")`
impl Default for LeidenParams {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            beta: 0.01,
            n_iterations: 2,
            refine: true,
            seed: 42,
            local_moving: LocalMoving::Queue,
        }
    }
}

////////////////
// LevelGraph //
////////////////

/// Read access to the adjacency of one level of the hierarchy.
///
/// Level 0 is the caller's graph relabelled into BFS order, every later level
/// an aggregate; both are an [`AggGraph`]. [`SparseGraph`] implements it too,
/// for the BFS and the relabelling themselves.
trait LevelGraph: Sync {
    /// Edge weight type
    type W: Copy + Zero + Send + Sync + ToPrimitive;

    /// Number of nodes in this level
    fn n(&self) -> usize;

    /// Neighbours and weights of node `i`
    fn row(&self, i: usize) -> (&[u32], &[Self::W]);
}

impl<T> LevelGraph for SparseGraph<T>
where
    T: BixverseFloat + Send + Sync + std::iter::Sum,
{
    type W = T;

    #[inline]
    fn n(&self) -> usize {
        self.get_node_number()
    }

    #[inline]
    fn row(&self, i: usize) -> (&[u32], &[T]) {
        self.get_neighbours_raw(i)
    }
}

/// Symmetric CSR for one level of the hierarchy.
///
/// Level 0 is the caller's graph relabelled into BFS order, weights in the
/// caller's type. Aggregated levels hold f64 weights and no self-loops: local
/// moving and refinement ignore them, and the node weights carry the full
/// degree separately.
struct AggGraph<W> {
    /// Row offsets, length n + 1
    indptr: Vec<usize>,
    /// Column indices
    indices: Vec<u32>,
    /// Edge weights
    weights: Vec<W>,
}

impl<W: Copy + Zero + Send + Sync + ToPrimitive> LevelGraph for AggGraph<W> {
    type W = W;

    #[inline]
    fn n(&self) -> usize {
        self.indptr.len() - 1
    }

    #[inline]
    fn row(&self, i: usize) -> (&[u32], &[W]) {
        let (s, e) = (self.indptr[i], self.indptr[i + 1]);
        (&self.indices[s..e], &self.weights[s..e])
    }
}

/////////////
// Helpers //
/////////////

/// Weight to f64. `ToPrimitive` on f32/f64 cannot fail.
///
/// ### Params
///
/// * `w` - Edge weight
///
/// ### Returns
///
/// The weight as f64
#[inline(always)]
fn wf<W: ToPrimitive>(w: W) -> f64 {
    w.to_f64().expect("float to f64 cannot fail")
}

/// SplitMix64 finaliser, used to derive independent per-cluster seeds.
///
/// ### Params
///
/// * `x` - Value to mix
///
/// ### Returns
///
/// Mixed 64-bit value
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Derive a seed from a base seed and a list of indices.
///
/// ### Params
///
/// * `seed` - Base seed
/// * `parts` - Indices identifying the stream (iteration, level, cluster)
///
/// ### Returns
///
/// Derived seed
fn derive_seed(seed: u64, parts: &[u64]) -> u64 {
    parts
        .iter()
        .fold(splitmix64(seed), |acc, &p| splitmix64(acc ^ p))
}

/// Renumber labels to `0..k` in order of first appearance.
///
/// ### Params
///
/// * `labels` - Labels, each `< labels.len()`; renumbered in place
///
/// ### Returns
///
/// Number of distinct labels
fn reindex(labels: &mut [u32]) -> usize {
    let mut map = vec![u32::MAX; labels.len()];
    let mut next = 0u32;
    for l in labels.iter_mut() {
        let m = &mut map[*l as usize];
        if *m == u32::MAX {
            *m = next;
            next += 1;
        }
        *l = *m;
    }
    next as usize
}

/// Group nodes by label (counting sort), members in increasing node order.
///
/// ### Params
///
/// * `labels` - Label per node, contiguous in `0..k`
/// * `k` - Number of labels
///
/// ### Returns
///
/// `(offsets, members)`: members of label `c` are
/// `members[offsets[c]..offsets[c + 1]]`
fn group_by_label(labels: &[u32], k: usize) -> (Vec<usize>, Vec<u32>) {
    let mut offsets = vec![0usize; k + 1];
    for &l in labels {
        offsets[l as usize + 1] += 1;
    }
    for c in 0..k {
        offsets[c + 1] += offsets[c];
    }
    let mut cursor = offsets.clone();
    let mut members = vec![0u32; labels.len()];
    for (v, &l) in labels.iter().enumerate() {
        members[cursor[l as usize]] = v as u32;
        cursor[l as usize] += 1;
    }
    (offsets, members)
}

/// Modularity of a partition.
///
/// `Q = 1/2m * sum_c (e_c - gamma * K_c^2 / 2m)`, with `e_c` twice the
/// internal edge weight and `K_c` the summed degree of cluster `c`.
/// Self-loops count once per CSR entry, as in igraph.
///
/// ### Params
///
/// * `graph` - Undirected [`SparseGraph`]
/// * `labels` - Community per node
/// * `resolution` - Resolution parameter (gamma)
///
/// ### Returns
///
/// The modularity
pub fn modularity<T>(graph: &SparseGraph<T>, labels: &[usize], resolution: f64) -> f64
where
    T: BixverseFloat + Send + Sync + std::iter::Sum,
{
    let n = graph.get_node_number();
    let k = labels.iter().copied().max().map_or(0, |m| m + 1);
    let mut internal = 0.0f64;
    let mut two_m = 0.0f64;
    let mut cluster_w = vec![0.0f64; k];
    for v in 0..n {
        let (nbrs, ws) = graph.get_neighbours_raw(v);
        let mut deg = 0.0;
        for (&u, &w) in nbrs.iter().zip(ws) {
            let w = wf(w);
            deg += w;
            if labels[u as usize] == labels[v] {
                internal += w;
            }
        }
        two_m += deg;
        cluster_w[labels[v]] += deg;
    }
    if two_m == 0.0 {
        return 0.0;
    }
    let penalty: f64 = cluster_w.iter().map(|&c| c * c).sum::<f64>() * resolution / two_m;
    (internal - penalty) / two_m
}

///////////////////
// Local moving //
///////////////////

/// Fast local moving (Traag et al. 2019, Alg. A.2 `MoveNodesFast`).
///
/// Nodes sit in a queue, initially in shuffled or rotated order. A node takes the
/// neighbouring cluster (or an empty one) with the strictly best gain over
/// staying; when it moves, its stable neighbours outside the new cluster are
/// re-queued. Self-loops are skipped in the scan.
///
/// ### Params
///
/// * `g` - Graph of this level
/// * `node_w` - Node weights (degrees at level 0, summed degrees after)
/// * `membership` - Initial partition, labels `< n`; updated and reindexed
/// * `res` - Resolution divided by `2m` of the level-0 graph
/// * `max_evals` - Cap on node evaluations (Louvain safeguard)
/// * `shuffle` - Visit nodes in random order instead of rotated node order
/// * `rng` - Random number generator for the initial order
///
/// ### Returns
///
/// `(number of clusters, whether any node moved)`
fn move_nodes<G: LevelGraph>(
    g: &G,
    node_w: &[f64],
    membership: &mut [u32],
    res: f64,
    max_evals: usize,
    shuffle: bool,
    rng: &mut StdRng,
) -> (usize, bool) {
    let n = g.n();

    // Without shuffling, nodes are visited in node order from a random start.
    // Node order is BFS order at level 0 and follows it at coarser levels, so
    // the scans stay cache-local and need about half the evaluations. Leiden
    // only: without refinement, the ordered sweep snowballs clusters and costs
    // Louvain modularity.
    let queue_order: Vec<u32> = if shuffle {
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.shuffle(rng);
        order
    } else {
        let start = if n > 0 { rng.random_range(0..n) } else { 0 };
        (start..n).chain(0..start).map(|v| v as u32).collect()
    };
    let mut changed = false;
    let mut queue: VecDeque<u32> = queue_order.into();
    let mut stable = vec![false; n];

    let mut cluster_w = vec![0.0f64; n];
    let mut cluster_n = vec![0u32; n];
    for v in 0..n {
        let c = membership[v] as usize;
        cluster_w[c] += node_w[v];
        cluster_n[c] += 1;
    }
    let mut empty: Vec<u32> = (0..n as u32)
        .filter(|&c| cluster_n[c as usize] == 0)
        .collect();

    let mut w_to = vec![0.0f64; n];
    let mut added = vec![false; n];
    let mut nbr_clusters: Vec<u32> = Vec::with_capacity(64);

    let mut evals = 0usize;
    while let Some(v) = queue.pop_front() {
        if evals >= max_evals {
            break;
        }
        evals += 1;

        let v = v as usize;
        let cur = membership[v];
        let kv = node_w[v];

        cluster_w[cur as usize] -= kv;
        cluster_n[cur as usize] -= 1;
        if cluster_n[cur as usize] == 0 {
            empty.push(cur);
        }

        // an empty cluster is always a candidate (the node on its own)
        let e = *empty
            .last()
            .expect("node just left a cluster, so one is free or v is alone");
        nbr_clusters.push(e);
        added[e as usize] = true;

        let (nbrs, ws) = g.row(v);
        for (&u, &w) in nbrs.iter().zip(ws) {
            if u as usize == v {
                continue;
            }
            let c = membership[u as usize];
            if !added[c as usize] {
                added[c as usize] = true;
                nbr_clusters.push(c);
            }
            w_to[c as usize] += wf(w);
        }

        let mut best = cur;
        let mut best_gain = w_to[cur as usize] - kv * cluster_w[cur as usize] * res;
        for &c in &nbr_clusters {
            let gain = w_to[c as usize] - kv * cluster_w[c as usize] * res;
            // strictly improving only, needed for convergence
            if gain > best_gain {
                best = c;
                best_gain = gain;
            }
            w_to[c as usize] = 0.0;
            added[c as usize] = false;
        }
        nbr_clusters.clear();

        cluster_w[best as usize] += kv;
        cluster_n[best as usize] += 1;
        if empty.last() == Some(&best) {
            empty.pop();
        }

        stable[v] = true;
        if best != cur {
            changed = true;
            membership[v] = best;
            for &u in nbrs {
                let u = u as usize;
                if stable[u] && membership[u] != best {
                    stable[u] = false;
                    queue.push_back(u as u32);
                }
            }
        }
    }

    (reindex(membership), changed)
}

///////////////////////////
// Parallel local moving //
///////////////////////////

/// Open-addressing map from cluster id to edge weight, for one row.
///
/// Sized to the row rather than to the cluster count, so the per-thread
/// scratch does not grow with n (an n-long f64 array per thread is 66 MB on
/// ten threads at 829k nodes).
#[derive(Default)]
struct ClusterTable {
    /// Cluster id per slot, `u32::MAX` when free
    keys: Vec<u32>,
    /// Summed edge weight per slot
    vals: Vec<f64>,
    /// Occupied slots in insertion order
    used: Vec<u32>,
}

impl ClusterTable {
    /// Make room for a row of `deg` entries; the table must be empty.
    ///
    /// ### Params
    ///
    /// * `deg` - Entries in the row
    #[inline]
    fn reserve(&mut self, deg: usize) {
        let cap = (2 * deg).next_power_of_two().max(16);
        if self.keys.len() < cap {
            self.keys = vec![u32::MAX; cap];
            self.vals = vec![0.0; cap];
        }
    }

    /// Slot of `c`, or the free slot where it goes.
    ///
    /// ### Params
    ///
    /// * `c` - Cluster id
    ///
    /// ### Returns
    ///
    /// The slot index
    #[inline]
    fn slot(&self, c: u32) -> usize {
        let mask = self.keys.len() - 1;
        let mut i = (c.wrapping_mul(0x9E37_79B1) as usize) & mask;
        while self.keys[i] != c && self.keys[i] != u32::MAX {
            i = (i + 1) & mask;
        }
        i
    }

    /// Add `w` to cluster `c`.
    ///
    /// ### Params
    ///
    /// * `c` - Cluster id
    /// * `w` - Edge weight
    #[inline]
    fn add(&mut self, c: u32, w: f64) {
        let i = self.slot(c);
        if self.keys[i] == u32::MAX {
            self.keys[i] = c;
            self.vals[i] = w;
            self.used.push(i as u32);
        } else {
            self.vals[i] += w;
        }
    }

    /// Summed weight to cluster `c`, 0 if absent.
    ///
    /// ### Params
    ///
    /// * `c` - Cluster id
    ///
    /// ### Returns
    ///
    /// The weight
    #[inline]
    fn get(&self, c: u32) -> f64 {
        let i = self.slot(c);
        if self.keys[i] == c { self.vals[i] } else { 0.0 }
    }

    /// Empty the table.
    #[inline]
    fn clear(&mut self) {
        for &i in &self.used {
            self.keys[i as usize] = u32::MAX;
        }
        self.used.clear();
    }
}

thread_local! {
    /// Per-thread [`ClusterTable`] for [`best_move`]. Empty between uses.
    static MOVE_SCRATCH: RefCell<ClusterTable> = RefCell::new(ClusterTable::default());
}

/// A move decided by [`best_move`].
#[derive(Clone, Copy, Debug)]
struct Move {
    /// Node
    v: u32,
    /// Target cluster, or [`NEW_CLUSTER`]
    to: u32,
    /// Edge weight from `v` to the target (0 for a new cluster)
    w_to: f64,
    /// Edge weight from `v` to its own cluster, `v` excluded
    w_from: f64,
}

/// Local moving with the strategy in `mode`.
///
/// Levels below [`LEIDEN_PAR_MIN_NODES`] nodes always take the queue: they
/// are cheap and gain nothing from a fork.
///
/// ### Params
///
/// * `g` - Graph of this level
/// * `node_w` - Node weights
/// * `membership` - Initial partition, labels `< n`; updated and reindexed
/// * `res` - Resolution divided by `2m`
/// * `max_evals` - Cap on node evaluations
/// * `shuffle` - Queue only: random instead of rotated node order
/// * `mode` - The [`LocalMoving`] strategy
/// * `rng` - Random number generator for the node orders
///
/// ### Returns
///
/// `(number of clusters, whether any node moved)`
#[allow(clippy::too_many_arguments)]
fn local_move<G: LevelGraph>(
    g: &G,
    node_w: &[f64],
    membership: &mut [u32],
    res: f64,
    max_evals: usize,
    shuffle: bool,
    mode: LocalMoving,
    rng: &mut StdRng,
) -> (usize, bool) {
    if g.n() < LEIDEN_PAR_MIN_NODES {
        return move_nodes(g, node_w, membership, res, max_evals, shuffle, rng);
    }
    match mode {
        LocalMoving::Queue => move_nodes(g, node_w, membership, res, max_evals, shuffle, rng),
        LocalMoving::Coloured => move_nodes_coloured(g, node_w, membership, res, max_evals, rng),
    }
}

/// Best move of `v` against fixed cluster totals.
///
/// Candidates are the clusters of `v`'s neighbours and, if `v` is not alone,
/// an empty cluster at gain 0. Only a strict improvement over staying counts;
/// ties between candidates go to the lower cluster id, and an empty cluster
/// is only taken when no neighbouring cluster beats staying.
///
/// ### Params
///
/// * `g` - Graph of this level
/// * `v` - The node
/// * `membership` - Cluster per node
/// * `cluster_w` - Summed node weight per cluster, `v` included
/// * `cluster_n` - Nodes per cluster
/// * `kv` - Node weight of `v`
/// * `res` - Resolution divided by `2m`
/// * `table` - Scratch, empty
///
/// ### Returns
///
/// The move, or `None` to stay
#[allow(clippy::too_many_arguments)]
#[inline]
fn best_move<G: LevelGraph>(
    g: &G,
    v: u32,
    membership: &[u32],
    cluster_w: &[f64],
    cluster_n: &[u32],
    kv: f64,
    res: f64,
    table: &mut ClusterTable,
) -> Option<Move> {
    let (nbrs, ws) = g.row(v as usize);
    table.reserve(nbrs.len());
    for (&u, &w) in nbrs.iter().zip(ws) {
        if u != v {
            table.add(membership[u as usize], wf(w));
        }
    }
    let own = membership[v as usize];
    let w_from = table.get(own);
    let stay = w_from - kv * (cluster_w[own as usize] - kv) * res;

    let mut best = (own, stay);
    let mut w_best = 0.0;
    for &i in &table.used {
        let c = table.keys[i as usize];
        if c == own {
            continue;
        }
        let gain = table.vals[i as usize] - kv * cluster_w[c as usize] * res;
        if gain > best.1 || (gain == best.1 && best.0 != own && c < best.0) {
            best = (c, gain);
            w_best = table.vals[i as usize];
        }
    }
    if best.0 == own && cluster_n[own as usize] > 1 && 0.0 > stay {
        best = (NEW_CLUSTER, 0.0);
        w_best = 0.0;
    }
    table.clear();
    (best.0 != own).then_some(Move {
        v,
        to: best.0,
        w_to: w_best,
        w_from,
    })
}

/// Decide the moves of `nodes` in parallel against fixed cluster totals.
///
/// ### Params
///
/// * `g` - Graph of this level
/// * `nodes` - Nodes to evaluate, sorted
/// * `membership` - Cluster per node
/// * `cluster_w` - Summed node weight per cluster
/// * `cluster_n` - Nodes per cluster
/// * `node_w` - Node weights
/// * `res` - Resolution divided by `2m`
///
/// ### Returns
///
/// The moves, in the order of `nodes`
fn decide_moves<G: LevelGraph>(
    g: &G,
    nodes: &[u32],
    membership: &[u32],
    cluster_w: &[f64],
    cluster_n: &[u32],
    node_w: &[f64],
    res: f64,
) -> Vec<Move> {
    let decide = |&v: &u32| {
        MOVE_SCRATCH.with(|s| {
            best_move(
                g,
                v,
                membership,
                cluster_w,
                cluster_n,
                node_w[v as usize],
                res,
                &mut s.borrow_mut(),
            )
        })
    };
    nodes
        .par_iter()
        .with_min_len(PAR_MOVE_MIN_LEN)
        .filter_map(decide)
        .collect()
}

/// Flag the neighbours of the moved nodes that now sit in another cluster.
///
/// ### Params
///
/// * `g` - Graph of this level
/// * `moved` - Nodes that moved
/// * `membership` - Cluster per node, after the moves
/// * `pending` - Pending flag per node
///
/// ### Returns
///
/// The nodes whose flag this call set, in no particular order
fn flag_neighbours<G: LevelGraph>(
    g: &G,
    moved: &[u32],
    membership: &[u32],
    pending: &[AtomicBool],
) -> Vec<u32> {
    // load first: most neighbours are already pending, and a swap would
    // write their cache line anyway
    let flag = |&v: &u32| {
        let c = membership[v as usize];
        g.row(v as usize).0.iter().copied().filter(move |&u| {
            let p = &pending[u as usize];
            membership[u as usize] != c && !p.load(Relaxed) && !p.swap(true, Relaxed)
        })
    };
    moved
        .par_iter()
        .with_min_len(PAR_MOVE_MIN_LEN)
        .flat_map_iter(flag)
        .collect()
}

/// Take a pending list: sort it (so the result does not depend on the order
/// in which the flags were set) and clear the flags.
///
/// ### Params
///
/// * `list` - Pending nodes; emptied
/// * `pending` - Pending flag per node
///
/// ### Returns
///
/// The nodes, sorted
fn take_pending(list: &mut Vec<u32>, pending: &[AtomicBool]) -> Vec<u32> {
    let mut nodes = std::mem::take(list);
    nodes.par_sort_unstable();
    nodes
        .par_iter()
        .with_min_len(PAR_MOVE_MIN_LEN)
        .for_each(|&v| pending[v as usize].store(false, Relaxed));
    nodes
}

/// Summed node weight and node count per cluster.
///
/// ### Params
///
/// * `membership` - Cluster per node, labels `< n`
/// * `node_w` - Node weights
///
/// ### Returns
///
/// `(cluster weights, cluster sizes, empty clusters in descending id order)`
fn cluster_totals(membership: &[u32], node_w: &[f64]) -> (Vec<f64>, Vec<u32>, Vec<u32>) {
    let n = membership.len();
    let mut cluster_w = vec![0.0f64; n];
    let mut cluster_n = vec![0u32; n];
    for (v, &c) in membership.iter().enumerate() {
        cluster_w[c as usize] += node_w[v];
        cluster_n[c as usize] += 1;
    }
    let empty = (0..n as u32)
        .rev()
        .filter(|&c| cluster_n[c as usize] == 0)
        .collect();
    (cluster_w, cluster_n, empty)
}

/// Greedy first-fit colouring in the given order.
///
/// The colours of a node's neighbours below 64 go into one `u64` mask, so the
/// first free colour is a `trailing_ones`; only a node whose neighbours hold
/// all of 0..64 rescans with a full flag array. kNN graphs need far fewer
/// colours (829k harmony sNN: 42).
///
/// ### Params
///
/// * `g` - Graph of this level
/// * `order` - Visiting order
///
/// ### Returns
///
/// `(colour classes, each sorted by node id; colour per node)`
fn greedy_colouring<G: LevelGraph>(g: &G, order: &[u32]) -> (Vec<Vec<u32>>, Vec<u32>) {
    let n = g.n();
    let mut colour = vec![u32::MAX; n];
    let mut used: Vec<bool> = Vec::new();
    let mut classes: Vec<Vec<u32>> = Vec::new();
    for &v in order {
        let nbrs = g.row(v as usize).0;
        let mut mask = 0u64;
        for &u in nbrs {
            let c = colour[u as usize];
            if c < 64 {
                mask |= 1 << c;
            }
        }
        let c = if mask != u64::MAX {
            mask.trailing_ones() as usize
        } else {
            used.resize(classes.len(), false);
            for &u in nbrs {
                let cu = colour[u as usize];
                if cu != u32::MAX {
                    used[cu as usize] = true;
                }
            }
            let c = used.iter().position(|&x| !x).unwrap_or(used.len());
            used.iter_mut().for_each(|x| *x = false);
            c
        };
        if c == classes.len() {
            classes.push(Vec::new());
        }
        colour[v as usize] = c as u32;
        classes[c].push(v);
    }
    classes.par_iter_mut().for_each(|c| c.sort_unstable());
    (classes, colour)
}

/// Greedy colouring in node order from a seeded start.
///
/// At level 0 node order is BFS order, which keeps the scan cache-local.
///
/// ### Params
///
/// * `g` - Graph of this level
/// * `rng` - Random number generator for the start
///
/// ### Returns
///
/// `(colour classes, each sorted by node id; colour per node)`
fn seeded_colouring<G: LevelGraph>(g: &G, rng: &mut StdRng) -> (Vec<Vec<u32>>, Vec<u32>) {
    let n = g.n();
    let start = rng.random_range(0..n) as u32;
    let order: Vec<u32> = (start..n as u32).chain(0..start).collect();
    greedy_colouring(g, &order)
}

/// Local moving over greedy colour classes, see [`LocalMoving::Coloured`].
///
/// Colours come from [`seeded_colouring`]. Each class keeps a list of its
/// pending nodes; the classes are swept in turn until none is pending, so a
/// sweep costs what its pending nodes cost.
///
/// ### Params
///
/// * `g` - Graph of this level
/// * `node_w` - Node weights
/// * `membership` - Initial partition, labels `< n`; updated and reindexed
/// * `res` - Resolution divided by `2m`
/// * `max_evals` - Cap on node evaluations
/// * `rng` - Random number generator for the colouring start
///
/// ### Returns
///
/// `(number of clusters, whether any node moved)`
///
/// ### References
///
/// Lu, Halappanavar & Kalyanaraman, Parallel Computing, 2015 (colouring);
/// Sahu, arXiv:2312.13936, 2023 (GVE-Leiden, pruning)
fn move_nodes_coloured<G: LevelGraph>(
    g: &G,
    node_w: &[f64],
    membership: &mut [u32],
    res: f64,
    max_evals: usize,
    rng: &mut StdRng,
) -> (usize, bool) {
    let n = g.n();
    let (classes, colour) = seeded_colouring(g, rng);

    let (mut cluster_w, mut cluster_n, mut empty) = cluster_totals(membership, node_w);
    let pending: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(true)).collect();
    let mut lists = classes;
    let mut moved: Vec<u32> = Vec::new();
    let mut changed = false;
    let mut evals = 0usize;

    'sweeps: for _ in 0..PAR_MOVE_MAX_ROUNDS {
        if lists.iter().all(|l| l.is_empty()) {
            break;
        }
        for c in 0..lists.len() {
            if lists[c].is_empty() {
                continue;
            }
            let nodes = take_pending(&mut lists[c], &pending);
            evals += nodes.len();
            let moves = decide_moves(g, &nodes, membership, &cluster_w, &cluster_n, node_w, res);
            moved.clear();
            for m in moves {
                let (v, kv) = (m.v as usize, node_w[m.v as usize]);
                let from = membership[v] as usize;
                let stay = m.w_from - kv * (cluster_w[from] - kv) * res;
                let go = if m.to == NEW_CLUSTER {
                    0.0
                } else {
                    m.w_to - kv * cluster_w[m.to as usize] * res
                };
                // no longer an improvement, or already alone
                if go <= stay || (m.to == NEW_CLUSTER && cluster_n[from] == 1) {
                    continue;
                }
                let to = if m.to == NEW_CLUSTER {
                    empty
                        .pop()
                        .expect("a cluster with two or more nodes leaves one id free")
                } else {
                    m.to
                } as usize;
                cluster_w[from] -= kv;
                cluster_n[from] -= 1;
                if cluster_n[from] == 0 {
                    empty.push(from as u32);
                }
                cluster_w[to] += kv;
                cluster_n[to] += 1;
                membership[v] = to as u32;
                moved.push(v as u32);
            }
            changed |= !moved.is_empty();
            for u in flag_neighbours(g, &moved, membership, &pending) {
                lists[colour[u as usize] as usize].push(u);
            }
            if evals >= max_evals {
                break 'sweeps;
            }
        }
    }

    (reindex(membership), changed)
}

////////////////
// Refinement //
////////////////

/// Per-thread scratch for refining one cluster, indexed by local position.
///
/// ### Fields
///
/// * `cluster_w` - Summed node weight per refined sub-cluster
/// * `ext` - Edge weight from a sub-cluster to the rest of its cluster
/// * `w_to` - Edge weight from the current node to each sub-cluster
/// * `added` - Membership flags for `cands`
/// * `non_singleton` - Sub-cluster has already absorbed a node
/// * `cands` - Candidate sub-clusters for the current node
/// * `cum` - Cumulative sampling weights aligned with `cands`
/// * `gains` - Gains aligned with `cands` (`NAN` when not eligible)
#[derive(Default)]
struct RefineScratch {
    cluster_w: Vec<f64>,
    ext: Vec<f64>,
    w_to: Vec<f64>,
    added: Vec<bool>,
    non_singleton: Vec<bool>,
    cands: Vec<u32>,
    cum: Vec<f64>,
    gains: Vec<f64>,
}

impl RefineScratch {
    /// Reset the scratch for a cluster of `s` nodes.
    ///
    /// ### Params
    ///
    /// * `s` - Number of nodes in the cluster
    fn reset(&mut self, s: usize) {
        for v in [&mut self.cluster_w, &mut self.ext, &mut self.w_to] {
            v.clear();
            v.resize(s, 0.0);
        }
        for v in [&mut self.added, &mut self.non_singleton] {
            v.clear();
            v.resize(s, false);
        }
    }
}

/// Refine one cluster by randomised constrained merging (Traag et al. 2019,
/// Alg. A.2 `MergeNodesSubset`).
///
/// Every node starts as its own sub-cluster. Each node is visited once in a
/// random order; if it is still a singleton and well connected to the rest of
/// the cluster, it joins a well-connected sub-cluster with non-negative gain,
/// drawn with probability proportional to `exp(gain / beta)`. Staying put has
/// gain 0 and is always a candidate.
///
/// ### Params
///
/// * `g` - Graph of this level
/// * `node_w` - Node weights
/// * `membership` - Unrefined partition of this level
/// * `local_idx` - Position of each node within its cluster's member list
/// * `members` - Nodes of this cluster, in increasing order
/// * `cluster` - Label of this cluster in `membership`
/// * `res` - Resolution divided by `2m`
/// * `beta` - Randomness of the merges
/// * `seed` - Seed for this cluster
/// * `sc` - Scratch buffers
///
/// ### Returns
///
/// `(local sub-cluster per member, number of sub-clusters)`, labels
/// contiguous in order of first appearance
#[allow(clippy::too_many_arguments)]
fn refine_cluster<G: LevelGraph>(
    g: &G,
    node_w: &[f64],
    membership: &[u32],
    local_idx: &[u32],
    members: &[u32],
    cluster: u32,
    res: f64,
    beta: f64,
    seed: u64,
    sc: &mut RefineScratch,
) -> (Vec<u32>, usize) {
    let s = members.len();
    let mut label: Vec<u32> = (0..s as u32).collect();
    if s == 1 {
        return (label, 1);
    }
    sc.reset(s);

    let mut total_w = 0.0f64;
    for (i, &v) in members.iter().enumerate() {
        let v = v as usize;
        sc.cluster_w[i] = node_w[v];
        total_w += node_w[v];
        let (nbrs, ws) = g.row(v);
        for (&u, &w) in nbrs.iter().zip(ws) {
            if u as usize != v && membership[u as usize] == cluster {
                sc.ext[i] += wf(w);
            }
        }
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut order: Vec<u32> = (0..s as u32).collect();
    order.shuffle(&mut rng);

    for &i in &order {
        let i = i as usize;
        let cur = label[i] as usize;
        let kv = node_w[members[i] as usize];
        let well_connected = |sc: &RefineScratch, c: usize| {
            sc.ext[c] >= sc.cluster_w[c] * (total_w - sc.cluster_w[c]) * res
        };

        if sc.non_singleton[cur] || !well_connected(sc, cur) {
            continue;
        }

        let ext_v = sc.ext[cur];
        sc.cluster_w[cur] = 0.0;

        sc.cands.push(cur as u32);
        sc.added[cur] = true;
        let (nbrs, ws) = g.row(members[i] as usize);
        for (&u, &w) in nbrs.iter().zip(ws) {
            let u = u as usize;
            if u == members[i] as usize || membership[u] != cluster {
                continue;
            }
            let c = label[local_idx[u] as usize] as usize;
            if !sc.added[c] {
                sc.added[c] = true;
                sc.cands.push(c as u32);
            }
            sc.w_to[c] += wf(w);
        }

        let mut best = cur;
        let mut max_gain = 0.0f64;
        for &c in &sc.cands {
            let c = c as usize;
            let gain = if well_connected(sc, c) {
                sc.w_to[c] - kv * sc.cluster_w[c] * res
            } else {
                f64::NAN
            };
            if gain > max_gain {
                max_gain = gain;
                best = c;
            }
            sc.gains.push(gain);
        }

        // shift by the max gain so exp() cannot overflow; same distribution
        let mut total = 0.0f64;
        for &gain in &sc.gains {
            if gain >= 0.0 {
                total += ((gain - max_gain) / beta).exp();
            }
            sc.cum.push(total);
        }
        let chosen = if total > 0.0 && total.is_finite() {
            let r = rng.random::<f64>() * total;
            let idx = sc.cum.partition_point(|&x| x <= r).min(sc.cands.len() - 1);
            sc.cands[idx] as usize
        } else {
            best
        };

        let w_chosen = sc.w_to[chosen];
        for &c in &sc.cands {
            sc.w_to[c as usize] = 0.0;
            sc.added[c as usize] = false;
        }
        sc.cands.clear();
        sc.gains.clear();
        sc.cum.clear();

        sc.cluster_w[chosen] += kv;
        if chosen != cur {
            // E(T + v, C - T - v) = E(T, C - T) + E(v, C - v) - 2 E(v, T)
            sc.ext[chosen] += ext_v - 2.0 * w_chosen;
            label[i] = chosen as u32;
            sc.non_singleton[chosen] = true;
        }
    }

    let k = reindex(&mut label);
    (label, k)
}

/// Refine every cluster of the unrefined partition.
///
/// Parallel over clusters when the level is large enough. Each cluster draws
/// from its own seeded RNG, and sub-cluster ids are laid out in cluster order,
/// so the result does not depend on the thread count.
///
/// ### Params
///
/// * `g` - Graph of this level
/// * `node_w` - Node weights
/// * `membership` - Unrefined partition, contiguous labels
/// * `n_clusters` - Number of clusters in `membership`
/// * `res` - Resolution divided by `2m`
/// * `beta` - Randomness of the merges
/// * `seed` - Seed for this iteration and level
///
/// ### Returns
///
/// `(refined partition, number of refined clusters)`
fn refine_partition<G: LevelGraph>(
    g: &G,
    node_w: &[f64],
    membership: &[u32],
    n_clusters: usize,
    res: f64,
    beta: f64,
    seed: u64,
) -> (Vec<u32>, usize) {
    let n = g.n();
    let (offsets, members) = group_by_label(membership, n_clusters);
    let mut local_idx = vec![0u32; n];
    for c in 0..n_clusters {
        for (i, &v) in members[offsets[c]..offsets[c + 1]].iter().enumerate() {
            local_idx[v as usize] = i as u32;
        }
    }

    let run = |sc: &mut RefineScratch, c: usize| {
        refine_cluster(
            g,
            node_w,
            membership,
            &local_idx,
            &members[offsets[c]..offsets[c + 1]],
            c as u32,
            res,
            beta,
            derive_seed(seed, &[c as u64]),
            sc,
        )
    };
    let per_cluster: Vec<(Vec<u32>, usize)> = if n >= LEIDEN_PAR_MIN_NODES {
        (0..n_clusters)
            .into_par_iter()
            .map_init(RefineScratch::default, run)
            .collect()
    } else {
        let mut sc = RefineScratch::default();
        (0..n_clusters).map(|c| run(&mut sc, c)).collect()
    };

    let mut refined = vec![0u32; n];
    let mut base = 0u32;
    for (c, (labels, k)) in per_cluster.into_iter().enumerate() {
        for (&v, &l) in members[offsets[c]..offsets[c + 1]].iter().zip(&labels) {
            refined[v as usize] = base + l;
        }
        base += k as u32;
    }
    (refined, base as usize)
}

/////////////////
// Aggregation //
/////////////////

/// Collapse each refined cluster into one node.
///
/// Parallel over refined clusters when the level is large enough, with a
/// dense per-thread accumulator cleared through its touched list. Rows come
/// out in member order, so the sums are deterministic.
///
/// ### Params
///
/// * `g` - Graph of this level
/// * `node_w` - Node weights
/// * `refined` - Refined partition, contiguous labels
/// * `n_refined` - Number of refined clusters
/// * `membership` - Unrefined partition; refines `refined`
///
/// ### Returns
///
/// `(aggregated graph, aggregated node weights, initial aggregated partition)`
fn aggregate<G: LevelGraph>(
    g: &G,
    node_w: &[f64],
    refined: &[u32],
    n_refined: usize,
    membership: &[u32],
) -> (AggGraph<f64>, Vec<f64>, Vec<u32>) {
    let (offsets, members) = group_by_label(refined, n_refined);

    let row = |(acc, seen, touched): &mut (Vec<f64>, Vec<bool>, Vec<u32>), c: usize| {
        if acc.len() < n_refined {
            acc.resize(n_refined, 0.0);
            seen.resize(n_refined, false);
        }
        let mut w_sum = 0.0;
        for &v in &members[offsets[c]..offsets[c + 1]] {
            w_sum += node_w[v as usize];
            let (nbrs, ws) = g.row(v as usize);
            for (&u, &w) in nbrs.iter().zip(ws) {
                let c2 = refined[u as usize];
                if c2 as usize == c {
                    continue;
                }
                if !seen[c2 as usize] {
                    seen[c2 as usize] = true;
                    touched.push(c2);
                }
                acc[c2 as usize] += wf(w);
            }
        }
        let mut idx = Vec::with_capacity(touched.len());
        let mut wts = Vec::with_capacity(touched.len());
        for &c2 in touched.iter() {
            idx.push(c2);
            wts.push(acc[c2 as usize]);
            acc[c2 as usize] = 0.0;
            seen[c2 as usize] = false;
        }
        touched.clear();
        (idx, wts, w_sum)
    };

    let rows: Vec<(Vec<u32>, Vec<f64>, f64)> = if g.n() >= LEIDEN_PAR_MIN_NODES {
        (0..n_refined)
            .into_par_iter()
            .map_init(|| (Vec::new(), Vec::new(), Vec::new()), row)
            .collect()
    } else {
        let mut scratch = (Vec::new(), Vec::new(), Vec::new());
        (0..n_refined).map(|c| row(&mut scratch, c)).collect()
    };

    let mut indptr = Vec::with_capacity(n_refined + 1);
    indptr.push(0usize);
    for r in &rows {
        indptr.push(indptr.last().unwrap() + r.0.len());
    }
    let nnz = *indptr.last().unwrap();
    let mut indices = Vec::with_capacity(nnz);
    let mut weights = Vec::with_capacity(nnz);
    let mut agg_w = Vec::with_capacity(n_refined);
    for (idx, wts, w_sum) in rows {
        indices.extend_from_slice(&idx);
        weights.extend_from_slice(&wts);
        agg_w.push(w_sum);
    }

    let agg_membership = (0..n_refined)
        .map(|c| membership[members[offsets[c]] as usize])
        .collect();

    (
        AggGraph {
            indptr,
            indices,
            weights,
        },
        agg_w,
        agg_membership,
    )
}

////////////////
// Reordering //
////////////////

/// Breadth-first node order, restarting at the lowest unvisited node for
/// each further component.
///
/// ### Params
///
/// * `g` - Graph to order
///
/// ### Returns
///
/// `order`, with `order[new] = old`
fn bfs_order<G: LevelGraph>(g: &G) -> Vec<u32> {
    let n = g.n();
    let mut seen = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for s in 0..n {
        if seen[s] {
            continue;
        }
        seen[s] = true;
        let mut head = order.len();
        order.push(s as u32);
        while head < order.len() {
            let v = order[head] as usize;
            head += 1;
            for &u in g.row(v).0 {
                if !seen[u as usize] {
                    seen[u as usize] = true;
                    order.push(u);
                }
            }
        }
    }
    order
}

/// Relabel a graph into a new node order.
///
/// ### Params
///
/// * `g` - Graph to relabel
/// * `order` - `order[new] = old`
/// * `pos` - Inverse of `order`, `pos[old] = new`
///
/// ### Returns
///
/// The relabelled graph, weights in the original type
fn permute_graph<G: LevelGraph>(g: &G, order: &[u32], pos: &[u32]) -> AggGraph<G::W> {
    let mut indptr = Vec::with_capacity(order.len() + 1);
    indptr.push(0usize);
    for &old in order {
        indptr.push(indptr.last().unwrap() + g.row(old as usize).0.len());
    }
    let nnz = *indptr.last().unwrap();
    let mut indices = vec![0u32; nnz];
    let mut weights = vec![G::W::zero(); nnz];

    // one disjoint (indices, weights) slice per new row, filled in parallel
    let mut rows = Vec::with_capacity(order.len());
    let (mut idx_rest, mut w_rest) = (indices.as_mut_slice(), weights.as_mut_slice());
    for r in 0..order.len() {
        let len = indptr[r + 1] - indptr[r];
        let (idx_row, idx_tail) = idx_rest.split_at_mut(len);
        let (w_row, w_tail) = w_rest.split_at_mut(len);
        rows.push((idx_row, w_row));
        idx_rest = idx_tail;
        w_rest = w_tail;
    }
    rows.par_iter_mut()
        .zip(order.par_iter())
        .for_each(|((idx_row, w_row), &old)| {
            let (nbrs, ws) = g.row(old as usize);
            for (d, &u) in idx_row.iter_mut().zip(nbrs) {
                *d = pos[u as usize];
            }
            w_row.copy_from_slice(ws);
        });

    AggGraph {
        indptr,
        indices,
        weights,
    }
}

//////////
// Core //
//////////

/// One full Leiden (or Louvain) iteration, starting from `membership`.
///
/// Local moving, refinement and aggregation per level, until local moving
/// leaves every node of a level in its own cluster.
///
/// ### Params
///
/// * `graph` - Level-0 graph
/// * `node_w` - Level-0 node weights (degrees)
/// * `membership` - Partition of the level-0 nodes; updated
/// * `res` - Resolution divided by `2m`
/// * `params` - The [`LeidenParams`]
/// * `iteration` - Index of this iteration, for seeding
/// * `max_evals` - Node-evaluation cap per local-moving phase, in units of n
/// * `multi_level` - `false` stops after the first local-moving phase
/// * `level0` - Local moving of level 0: `(graph, node weights, membership,
///   res, rng) -> (number of clusters, whether any node moved)`, membership
///   reindexed
///
/// ### Returns
///
/// Whether any node moved
#[allow(clippy::too_many_arguments)]
fn leiden_iteration<G, E, F>(
    graph: &G,
    node_w: &[f64],
    membership: &mut [u32],
    res: f64,
    params: &LeidenParams,
    iteration: u64,
    max_evals: usize,
    multi_level: bool,
    level0: &mut F,
) -> Result<bool, E>
where
    G: LevelGraph,
    F: FnMut(&G, &[f64], &mut [u32], f64, &mut StdRng) -> Result<(usize, bool), E>,
{
    let n = graph.n();
    let mut rng = StdRng::seed_from_u64(derive_seed(params.seed, &[iteration]));
    reindex(membership);

    let (n_clusters, mut changed) = level0(graph, node_w, membership, res, &mut rng)?;
    if !multi_level || n_clusters == n {
        return Ok(changed);
    }

    let (refined, n_refined) = refined_or_unrefined(
        graph, node_w, membership, n_clusters, res, params, iteration, 0,
    );
    let mut aggregate_node = refined.clone();
    let (mut g, mut w, mut agg_memb) = aggregate(graph, node_w, &refined, n_refined, membership);

    let mut level = 1u64;
    loop {
        let gn = g.n();
        let (nc, ch) = local_move(
            &g,
            &w,
            &mut agg_memb,
            res,
            max_evals.saturating_mul(gn),
            !params.refine,
            params.local_moving,
            &mut rng,
        );
        changed |= ch;
        for (m, &a) in membership.iter_mut().zip(&aggregate_node) {
            *m = agg_memb[a as usize];
        }
        if nc == gn {
            break;
        }

        let (refined, n_refined) =
            refined_or_unrefined(&g, &w, &agg_memb, nc, res, params, iteration, level);
        for a in aggregate_node.iter_mut() {
            *a = refined[*a as usize];
        }
        (g, w, agg_memb) = aggregate(&g, &w, &refined, n_refined, &agg_memb);
        level += 1;
    }

    Ok(changed)
}

/// Refined partition for Leiden, the unrefined one for Louvain.
///
/// As in igraph, a refinement that merged nothing falls back to the
/// unrefined partition, or the hierarchy would not shrink.
///
/// ### Params
///
/// * `g` - Graph of this level
/// * `node_w` - Node weights
/// * `membership` - Unrefined partition, contiguous labels
/// * `n_clusters` - Number of clusters in `membership`
/// * `res` - Resolution divided by `2m`
/// * `params` - The [`LeidenParams`]
/// * `iteration` - Iteration index, for seeding
/// * `level` - Level index, for seeding
///
/// ### Returns
///
/// `(partition to aggregate on, number of clusters in it)`
#[allow(clippy::too_many_arguments)]
fn refined_or_unrefined<G: LevelGraph>(
    g: &G,
    node_w: &[f64],
    membership: &[u32],
    n_clusters: usize,
    res: f64,
    params: &LeidenParams,
    iteration: u64,
    level: u64,
) -> (Vec<u32>, usize) {
    if params.refine {
        let seed = derive_seed(params.seed, &[iteration, level]);
        let (refined, k) =
            refine_partition(g, node_w, membership, n_clusters, res, params.beta, seed);
        if k < g.n() {
            return (refined, k);
        }
    }
    (membership.to_vec(), n_clusters)
}

/// Level-0 local moving on another backend (a GPU), for
/// [`community_detection_with`].
///
/// The engine hands over the BFS-relabelled level-0 graph once, then one
/// colouring per iteration. Coarser levels, refinement and aggregation stay on
/// the CPU.
pub trait LocalMover {
    /// Backend error
    type Error: From<MethodsErrors>;

    /// Take the level-0 graph. Called once, before the first
    /// [`LocalMover::move_nodes`].
    ///
    /// ### Params
    ///
    /// * `indptr` - Row offsets, length n + 1
    /// * `indices` - Column indices, symmetric, self-loops possible
    /// * `weights` - Edge weights
    /// * `node_w` - Node weights (weighted degree, self-loops included)
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or the backend's error
    fn prepare<W: ToPrimitive + Sync>(
        &mut self,
        indptr: &[usize],
        indices: &[u32],
        weights: &[W],
        node_w: &[f64],
    ) -> Result<(), Self::Error>;

    /// Coloured local moving on the prepared graph, see
    /// [`LocalMoving::Coloured`]. The apply rule may differ from the CPU's.
    ///
    /// ### Params
    ///
    /// * `membership` - Initial partition, labels `< n`; updated, labels
    ///   `< n` but not necessarily contiguous
    /// * `classes` - Colour classes, each sorted by node id
    /// * `colour` - Colour per node
    /// * `res` - Resolution divided by `2m`
    ///
    /// ### Returns
    ///
    /// Whether any node moved
    fn move_nodes(
        &mut self,
        membership: &mut [u32],
        classes: &[Vec<u32>],
        colour: &[u32],
        res: f64,
    ) -> Result<bool, Self::Error>;
}

/// Shared body of [`community_detection`] and [`community_detection_with`].
///
/// ### Params
///
/// * `graph` - Undirected [`SparseGraph`]
/// * `params` - The [`LeidenParams`]
/// * `max_evals` - Node-evaluation cap per local-moving phase, in units of n
/// * `multi_level` - `false` stops after the first local-moving phase
/// * `level0` - Level-0 local moving, see [`leiden_iteration`]
///
/// ### Returns
///
/// Community per node, `0` the largest
fn detect<T, E, F>(
    graph: &SparseGraph<T>,
    params: &LeidenParams,
    max_evals: usize,
    multi_level: bool,
    mut level0: F,
) -> Result<Vec<usize>, E>
where
    T: BixverseFloat + Send + Sync + std::iter::Sum,
    E: From<MethodsErrors> + Send,
    F: FnMut(&AggGraph<T>, &[f64], &mut [u32], f64, &mut StdRng) -> Result<(usize, bool), E> + Send,
{
    if graph.is_directed() {
        return Err(MethodsErrors::GraphDirectedError.into());
    }
    let n = graph.get_node_number();
    if n == 0 {
        return Ok(Vec::new());
    }
    assert!(n < u32::MAX as usize, "more than u32::MAX nodes");

    let order = bfs_order(graph);
    let mut pos = vec![0u32; n];
    for (new, &old) in order.iter().enumerate() {
        pos[old as usize] = new as u32;
    }
    let g = permute_graph(graph, &order, &pos);

    let node_w: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|v| g.row(v).1.iter().map(|&w| wf(w)).sum())
        .collect();
    let two_m: f64 = node_w.iter().sum();

    let mut membership: Vec<u32> = (0..n as u32).collect();
    if two_m > 0.0 {
        let res = params.resolution / two_m;
        // Run the loop on a pool thread: the parallel local-moving modes issue
        // thousands of short parallel calls, and each one from outside the pool
        // is an injection plus a blocking wake-up.
        rayon::scope(|_| {
            (0..params.n_iterations.max(1)).try_for_each(|it| {
                leiden_iteration(
                    &g,
                    &node_w,
                    &mut membership,
                    res,
                    params,
                    it as u64,
                    max_evals,
                    multi_level,
                    &mut level0,
                )
                .map(|_| ())
            })
        })?;
    }

    // back from BFS labels to the caller's node order
    let labels: Vec<usize> = pos
        .iter()
        .map(|&new| membership[new as usize] as usize)
        .collect();
    Ok(remap_communities_by_size(&labels))
}

/// Shared entry for Leiden and Louvain.
///
/// [`leiden_sparse_graph`] and [`louvain_sparse_graph`] are thin wrappers;
/// call this directly to cap the node evaluations or stop after one level.
///
/// ### Params
///
/// * `graph` - Undirected [`SparseGraph`]
/// * `params` - The [`LeidenParams`]
/// * `max_evals` - Node-evaluation cap per local-moving phase, in units of n
/// * `multi_level` - `false` stops after the first local-moving phase
///
/// ### Returns
///
/// Community per node, `0` the largest
pub fn community_detection<T>(
    graph: &SparseGraph<T>,
    params: &LeidenParams,
    max_evals: usize,
    multi_level: bool,
) -> Result<Vec<usize>, MethodsErrors>
where
    T: BixverseFloat + Send + Sync + std::iter::Sum,
{
    let (shuffle, mode) = (!params.refine, params.local_moving);
    detect(
        graph,
        params,
        max_evals,
        multi_level,
        |g, node_w, membership, res, rng| {
            let evals = max_evals.saturating_mul(g.n());
            Ok(local_move(
                g, node_w, membership, res, evals, shuffle, mode, rng,
            ))
        },
    )
}

/// [`community_detection`] with level-0 local moving on a [`LocalMover`].
///
/// Every iteration colours level 0 as [`LocalMoving::Coloured`] does and
/// hands the classes to `mover`; coarser levels use `params.local_moving` on
/// the CPU. `max_evals` caps the CPU levels only.
///
/// ### Params
///
/// * `graph` - Undirected [`SparseGraph`]
/// * `params` - The [`LeidenParams`]
/// * `max_evals` - Node-evaluation cap per CPU local-moving phase, in units
///   of n
/// * `multi_level` - `false` stops after the first local-moving phase
/// * `mover` - The level-0 [`LocalMover`]
///
/// ### Returns
///
/// Community per node, `0` the largest
pub fn community_detection_with<T, M>(
    graph: &SparseGraph<T>,
    params: &LeidenParams,
    max_evals: usize,
    multi_level: bool,
    mover: &mut M,
) -> Result<Vec<usize>, M::Error>
where
    T: BixverseFloat + Send + Sync + std::iter::Sum,
    M: LocalMover + Send,
    M::Error: Send,
{
    let mut prepared = false;
    detect(
        graph,
        params,
        max_evals,
        multi_level,
        |g, node_w, membership, res, rng| {
            if !prepared {
                mover.prepare(&g.indptr, &g.indices, &g.weights, node_w)?;
                prepared = true;
            }
            let (classes, colour) = seeded_colouring(g, rng);
            let changed = mover.move_nodes(membership, &classes, &colour, res)?;
            Ok((reindex(membership), changed))
        },
    )
}

//////////
// Main //
//////////

/// Leiden community detection on modularity.
///
/// Mirrors `igraph::cluster_leiden(objective_function = "modularity")`.
/// Communities are guaranteed to be connected.
///
/// ### Params
///
/// * `graph` - Undirected [`SparseGraph`]
/// * `params` - Optional [`LeidenParams`]; defaults match igraph
///
/// ### Returns
///
/// Community per node, relabelled by size (`0` the largest)
///
/// ### References
///
/// Traag, Waltman & van Eck, Scientific Reports, 2019
pub fn leiden_sparse_graph<T>(
    graph: &SparseGraph<T>,
    params: Option<LeidenParams>,
) -> Result<Vec<usize>, MethodsErrors>
where
    T: BixverseFloat + Send + Sync + std::iter::Sum,
{
    let params = params.unwrap_or_default();
    community_detection(graph, &params, usize::MAX, true)
}

/// Louvain community detection on modularity.
///
/// The Leiden engine with the refinement switched off and one iteration.
///
/// ### Params
///
/// * `graph` - Undirected [`SparseGraph`]
/// * `resolution` - Resolution parameter
/// * `max_iter` - Cap on node evaluations per local-moving phase, in units of
///   the level's node count. A safeguard for degenerate graphs; the queue
///   terminates on its own otherwise.
/// * `multi_level` - If `true`, run the full multi-level Louvain. If `false`,
///   run only the first local-moving phase (Phenograph / doubletdetection
///   behaviour).
/// * `seed` - Seed for reproducibility
///
/// ### Returns
///
/// Community per node, relabelled by size (`0` the largest)
///
/// ### References
///
/// Blondel et al., J. Stat. Mech., 2008
pub fn louvain_sparse_graph<T>(
    graph: &SparseGraph<T>,
    resolution: T,
    max_iter: usize,
    multi_level: bool,
    seed: usize,
) -> Result<Vec<usize>, MethodsErrors>
where
    T: BixverseFloat + Send + Sync + std::iter::Sum,
{
    let params = LeidenParams::new(wf(resolution), 0.01, 1, false, seed as u64);
    community_detection(graph, &params, max_iter, multi_level)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use bixverse_core::core::math::sparse::coo_to_csr;

    /// Zachary karate club, 0-based, from `igraph::make_graph("Zachary")`.
    const KARATE: [usize; 156] = [
        0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 10, 0, 11, 0, 12, 0, 13, 0, 17, 0, 19,
        0, 21, 0, 31, 1, 2, 1, 3, 1, 7, 1, 13, 1, 17, 1, 19, 1, 21, 1, 30, 2, 3, 2, 7, 2, 27, 2,
        28, 2, 32, 2, 9, 2, 8, 2, 13, 3, 7, 3, 12, 3, 13, 4, 6, 4, 10, 5, 6, 5, 10, 5, 16, 6, 16,
        8, 30, 8, 32, 8, 33, 9, 33, 13, 33, 14, 32, 14, 33, 15, 32, 15, 33, 18, 32, 18, 33, 19, 33,
        20, 32, 20, 33, 22, 32, 22, 33, 23, 25, 23, 27, 23, 32, 23, 33, 23, 29, 24, 25, 24, 27, 24,
        31, 25, 31, 26, 29, 26, 33, 27, 33, 28, 31, 28, 33, 29, 32, 29, 33, 30, 32, 30, 33, 31, 32,
        31, 33, 32, 33,
    ];

    fn build_undirected<T: BixverseFloat + BixverseNumeric + std::iter::Sum>(
        n: usize,
        edges: &[(usize, usize, T)],
        directed: bool,
    ) -> SparseGraph<T> {
        let mut rows = Vec::with_capacity(edges.len() * 2);
        let mut cols = Vec::with_capacity(edges.len() * 2);
        let mut vals = Vec::with_capacity(edges.len() * 2);
        for &(u, v, w) in edges {
            rows.extend([u, v]);
            cols.extend([v, u]);
            vals.extend([w, w]);
        }
        let csr = coo_to_csr(&rows.index_cast(), &cols.index_cast(), &vals, (n, n));
        SparseGraph::new(n, csr, directed)
    }

    fn karate() -> SparseGraph<f64> {
        let edges: Vec<(usize, usize, f64)> =
            KARATE.chunks_exact(2).map(|e| (e[0], e[1], 1.0)).collect();
        build_undirected(34, &edges, false)
    }

    /// Planted partition: `k` blocks of `size` nodes, `d_in` random edges per
    /// node inside its block and `d_out` outside, integer weights 1..=3.
    fn planted(
        k: usize,
        size: usize,
        d_in: usize,
        d_out: usize,
        seed: u64,
    ) -> Vec<(usize, usize, f64)> {
        let n = k * size;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut edges = Vec::new();
        for v in 0..n {
            let b = v / size;
            for _ in 0..d_in {
                let u = b * size + rng.random_range(0..size);
                if u != v {
                    edges.push((v, u, rng.random_range(1..=3) as f64));
                }
            }
            for _ in 0..d_out {
                let u = rng.random_range(0..n);
                if u / size != b {
                    edges.push((v, u, 1.0));
                }
            }
        }
        edges
    }

    fn n_communities(labels: &[usize]) -> usize {
        labels.iter().max().map_or(0, |m| m + 1)
    }

    /// Every community induces a connected subgraph.
    fn all_connected(graph: &SparseGraph<f64>, labels: &[usize]) -> bool {
        let n = graph.get_node_number();
        let mut seen = vec![false; n];
        let mut roots = vec![false; n_communities(labels)];
        for s in 0..n {
            if seen[s] {
                continue;
            }
            if roots[labels[s]] {
                return false;
            }
            roots[labels[s]] = true;
            let mut stack = vec![s];
            seen[s] = true;
            while let Some(v) = stack.pop() {
                for &u in graph.get_neighbours_raw(v).0 {
                    let u = u as usize;
                    if !seen[u] && labels[u] == labels[v] {
                        seen[u] = true;
                        stack.push(u);
                    }
                }
            }
        }
        true
    }

    /// Modularity matches `igraph::modularity` (values from Rscript).
    #[test]
    fn test_modularity_matches_igraph() {
        let g = karate();
        let labels: Vec<usize> = (0..34).map(|i| i % 2).collect();
        approx::assert_relative_eq!(
            modularity(&g, &labels, 1.0),
            -0.000082182774,
            epsilon = 1e-10
        );
        approx::assert_relative_eq!(
            modularity(&g, &labels, 0.5),
            0.249958908613,
            epsilon = 1e-10
        );
    }

    /// Leiden on karate reaches igraph's seed range (Rscript, 50 seeds:
    /// min 0.418803, median and max 0.419790).
    #[test]
    fn test_leiden_karate_modularity() {
        let g = karate();
        for seed in 0..10 {
            let p = LeidenParams {
                seed,
                ..Default::default()
            };
            let labels = leiden_sparse_graph(&g, Some(p)).unwrap();
            let q = modularity(&g, &labels, 1.0);
            assert!(q >= 0.418803 - 1e-6, "seed {seed}: modularity {q}");
            assert!(all_connected(&g, &labels));
        }
    }

    /// Louvain on karate stays within igraph's Louvain range (min 0.388560).
    #[test]
    fn test_louvain_karate_modularity() {
        let g = karate();
        for seed in 0..10 {
            let labels = louvain_sparse_graph(&g, 1.0, 50, true, seed).unwrap();
            let q = modularity(&g, &labels, 1.0);
            assert!(q >= 0.388560 - 1e-6, "seed {seed}: modularity {q}");
        }
    }

    /// Planted blocks are recovered exactly.
    #[test]
    fn test_leiden_recovers_planted_blocks() {
        let edges = planted(8, 200, 10, 1, 7);
        let g = build_undirected(1600, &edges, false);
        let labels = leiden_sparse_graph(&g, None).unwrap();
        assert_eq!(n_communities(&labels), 8);
        for v in 0..1600 {
            assert_eq!(labels[v], labels[(v / 200) * 200]);
        }
    }

    /// Same seed gives the same partition at 1 and 4 threads, on a graph large
    /// enough to take the parallel refinement and aggregation paths.
    #[test]
    fn test_leiden_thread_count_invariant() {
        let edges = planted(20, 400, 8, 3, 11);
        let g = build_undirected(8000, &edges, false);
        assert!(g.get_node_number() >= LEIDEN_PAR_MIN_NODES);
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| leiden_sparse_graph(&g, None).unwrap())
        };
        let one = run(1);
        assert_eq!(one, run(4));
        assert!(all_connected(&g, &one));
    }

    /// Leiden or Louvain with the given local-moving mode.
    fn run_mode(g: &SparseGraph<f64>, mode: LocalMoving, refine: bool, seed: u64) -> Vec<usize> {
        let p = LeidenParams {
            refine,
            n_iterations: if refine { 2 } else { 1 },
            seed,
            local_moving: mode,
            ..Default::default()
        };
        community_detection(g, &p, usize::MAX, true).unwrap()
    }

    /// Coloured gives the same partition at 1 and 4 threads, for Leiden and
    /// Louvain, and Leiden's communities stay connected.
    #[test]
    fn test_parallel_local_moving_thread_count_invariant() {
        let edges = planted(20, 400, 8, 3, 11);
        let g = build_undirected(8000, &edges, false);
        for refine in [true, false] {
            let run = |threads: usize| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| run_mode(&g, LocalMoving::Coloured, refine, 3))
            };
            let one = run(1);
            assert_eq!(one, run(4), "refine {refine}");
            if refine {
                assert!(all_connected(&g, &one));
            }
        }
    }

    /// Coloured recovers planted blocks exactly, on a graph large enough that
    /// it runs instead of the queue.
    #[test]
    fn test_parallel_local_moving_recovers_planted_blocks() {
        let edges = planted(8, 600, 10, 1, 7);
        let g = build_undirected(4800, &edges, false);
        assert!(g.get_node_number() >= LEIDEN_PAR_MIN_NODES);
        for refine in [true, false] {
            let labels = run_mode(&g, LocalMoving::Coloured, refine, 1);
            assert_eq!(n_communities(&labels), 8, "refine {refine}");
            for v in 0..4800 {
                assert_eq!(labels[v], labels[(v / 600) * 600], "refine {refine}");
            }
        }
    }

    /// The mask colouring equals a plain first-fit pass, including past 64
    /// colours (an 80-node clique on top of a planted graph).
    #[test]
    fn test_greedy_colouring_matches_first_fit() {
        let mut edges = planted(4, 100, 12, 2, 5);
        for u in 0..80 {
            for v in u + 1..80 {
                edges.push((u, v, 1.0));
            }
        }
        let g = build_undirected(400, &edges, false);
        let order: Vec<u32> = (150..400).chain(0..150).collect();
        let (classes, colour) = greedy_colouring(&g, &order);

        let mut want = vec![u32::MAX; 400];
        for &v in &order {
            let taken: Vec<u32> = g
                .row(v as usize)
                .0
                .iter()
                .map(|&u| want[u as usize])
                .collect();
            want[v as usize] = (0..).find(|c| !taken.contains(c)).unwrap();
        }
        assert_eq!(colour, want);
        assert!(classes.len() >= 80);
        for (c, members) in classes.iter().enumerate() {
            assert!(members.is_sorted());
            assert!(members.iter().all(|&v| colour[v as usize] == c as u32));
        }
    }

    /// Below `LEIDEN_PAR_MIN_NODES` Coloured takes the queue, so karate gives
    /// the queue's partition.
    #[test]
    fn test_parallel_local_moving_small_graph_uses_queue() {
        let g = karate();
        assert_eq!(
            run_mode(&g, LocalMoving::Coloured, true, 5),
            run_mode(&g, LocalMoving::Queue, true, 5)
        );
    }

    /// On a noisy planted partition Coloured lands within 1% of the queue's
    /// modularity, for Leiden and Louvain.
    #[test]
    fn test_parallel_local_moving_modularity_close_to_queue() {
        let edges = planted(20, 400, 8, 3, 11);
        let g = build_undirected(8000, &edges, false);
        for refine in [true, false] {
            let q_queue = modularity(&g, &run_mode(&g, LocalMoving::Queue, refine, 2), 1.0);
            let q = modularity(&g, &run_mode(&g, LocalMoving::Coloured, refine, 2), 1.0);
            assert!(
                q >= 0.99 * q_queue,
                "refine {refine}: {q} vs queue {q_queue}"
            );
        }
    }

    /// The same seed never gives a different partition.
    #[test]
    fn test_leiden_same_seed_same_labels() {
        let edges = planted(10, 100, 6, 3, 3);
        let g = build_undirected(1000, &edges, false);
        let p = LeidenParams {
            seed: 5,
            ..Default::default()
        };
        assert_eq!(
            leiden_sparse_graph(&g, Some(p)).unwrap(),
            leiden_sparse_graph(&g, Some(p)).unwrap()
        );
    }

    /// f32 and f64 weights give the same partition when the weights are exact
    /// in both, since all totals are held in f64.
    #[test]
    fn test_leiden_f32_matches_f64() {
        let edges = planted(6, 150, 6, 3, 9);
        let edges32: Vec<(usize, usize, f32)> =
            edges.iter().map(|&(u, v, w)| (u, v, w as f32)).collect();
        let g64 = build_undirected(900, &edges, false);
        let g32 = build_undirected(900, &edges32, false);
        assert_eq!(
            leiden_sparse_graph(&g64, None).unwrap(),
            leiden_sparse_graph(&g32, None).unwrap()
        );
    }

    /// Leiden communities are connected on a sparse noisy graph.
    #[test]
    fn test_leiden_communities_connected() {
        let edges = planted(30, 60, 3, 2, 21);
        let g = build_undirected(1800, &edges, false);
        for seed in 0..5 {
            let p = LeidenParams {
                seed,
                ..Default::default()
            };
            let labels = leiden_sparse_graph(&g, Some(p)).unwrap();
            assert!(all_connected(&g, &labels), "seed {seed}");
        }
    }

    #[test]
    fn test_leiden_directed_errors() {
        let edges: Vec<(usize, usize, f64)> =
            KARATE.chunks_exact(2).map(|e| (e[0], e[1], 1.0)).collect();
        let g = build_undirected(34, &edges, true);
        assert!(leiden_sparse_graph(&g, None).is_err());
    }

    #[test]
    fn test_leiden_empty_and_edgeless() {
        let g = build_undirected::<f64>(0, &[], false);
        assert!(leiden_sparse_graph(&g, None).unwrap().is_empty());
        let g = build_undirected::<f64>(3, &[], false);
        assert_eq!(leiden_sparse_graph(&g, None).unwrap().len(), 3);
    }
}
