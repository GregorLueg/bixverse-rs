//! Leiden and Louvain with level-0 local moving on the GPU.
//!
//! The CPU engine in `bixverse_methods::graph::leiden` colours level 0 and
//! hands the classes to [`LeidenGpuMover`]; coarser levels, refinement and
//! aggregation stay on the CPU. Per colour class the device runs decide,
//! resolve, link, accept, apply and mark; the host reads one move counter per
//! round.
//!
//! The apply rule is clust-libreville-rs's OrderFree rule rather than the
//! CPU's in-order re-check: a decided move is accepted against the class's
//! starting totals made pessimistic by every proposal of a lower node id into
//! its target and out of its source. Those prefixes are sums over sets, so
//! with exact integer totals the outcome does not depend on the order in which
//! threads run, and the partition is reproducible on one device.
//!
//! WGSL has no 64-bit integers, so the cluster totals are u32 fixed point at a
//! power-of-two scale chosen per graph, with `2m` at most 2^30 after scaling.
//! Sums and differences of the quantised node weights are exact, so a total
//! never drifts from its members. Gains are f32.

// kernel dialect: nested ifs because cubecl does not promise a short-circuit
// `&&`, and no `div_ceil` on expanded integers
#![allow(clippy::collapsible_if, clippy::manual_div_ceil)]

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;

use bixverse_methods::graph::leiden::{
    LeidenParams, LocalMover, LocalMoving, community_detection, community_detection_with,
};
use num_traits::ToPrimitive;

use crate::prelude::*;

////////////
// Consts //
////////////

/// Decide output: stay in the current cluster.
const STAY: u32 = u32::MAX - 1;

/// Decide output: move into an empty cluster (resolved to id `n + v`).
const NEW: u32 = u32::MAX;

/// End of a proposal list, and "no candidate" in the decide reduction.
const LIST_END: u32 = u32::MAX - 2;

/// Rows longer than this keep the whole run on the CPU. The decide kernel
/// costs `O(deg^2 / plane)` per lane; level-0 kNN graphs stay far below it
/// (829k harmony sNN: max degree 172).
const PLANE_DEG: usize = 512;

/// Vertices per cube of the plane kernels.
const PLANES_PER_CUBE: u32 = 4;

/// Threads per cube of the thread-per-entry kernels.
const BLOCK: u32 = 128;

/// Cap on the threads of one per-class launch. The kernels grid-stride over
/// the class's pending list, whose length only the device knows, so every
/// launch is sized for the whole class and capped here.
const GRID_CAP: usize = 16_384;

/// Safety cap on local-moving rounds, as on the CPU.
const MAX_ROUNDS: usize = 1000;

/// Fixed-point headroom: `2m` scales to at most `2^FX_BITS`, so totals plus
/// the rounding of up to `n / 2` stay below `2^31`.
const FX_BITS: i32 = 30;

/// Levels below this many nodes run on the CPU. Unmeasured placeholder until
/// the size sweep.
pub const LEIDEN_GPU_MIN_NODES: usize = 50_000;

/////////////
// Kernels //
/////////////

/// Snapshot the pending count of class `g` and reset it, so marks from later
/// classes append from slot 0 for the next round.
///
/// ### Params
///
/// * `acount` - Pending count per class
/// * `g` - Class index
/// * `cur` - Output, the snapshot
#[cube(launch_unchecked)]
fn lm_begin(acount: &mut Tensor<Atomic<u32>>, g: u32, cur: &mut Tensor<u32>) {
    if ABSOLUTE_POS == 0 {
        cur[0] = acount[g as usize].load();
        acount[g as usize].store(0u32);
    }
}

/// Zero a single counter.
///
/// ### Params
///
/// * `x` - The counter
#[cube(launch_unchecked)]
fn lm_zero(x: &mut Tensor<u32>) {
    if ABSOLUTE_POS == 0 {
        x[0] = 0u32;
    }
}

/// Fixed-point total and size of every cluster.
///
/// ### Params
///
/// * `comm` - Cluster per node
/// * `kint` - Fixed-point node weight
/// * `tot` - Fixed-point cluster totals, zeroed, length 2n
/// * `size` - Cluster sizes, zeroed, length 2n
/// * `n` - Number of nodes
#[cube(launch_unchecked)]
fn lm_init(
    comm: &Tensor<u32>,
    kint: &Tensor<u32>,
    tot: &mut Tensor<Atomic<u32>>,
    size: &mut Tensor<Atomic<u32>>,
    n: u32,
) {
    let v = ABSOLUTE_POS;
    if v < n as usize {
        let c = comm[v] as usize;
        tot[c].fetch_add(kint[v]);
        size[c].fetch_add(1u32);
    }
}

/// Best move of each pending node of a class, one plane per node, against the
/// totals frozen at the class start.
///
/// Lane `l` holds edge `start + l` (then `+ PLANE_DIM`, ...). Every lane sums
/// its edge's cluster over the whole row by broadcasting the plane's edges,
/// and only the first occurrence of a cluster counts as a candidate. The best
/// candidate is the highest gain, ties to the lower cluster id, and must beat
/// staying strictly; an empty cluster is taken when nothing beats staying and
/// staying is negative. The CPU's `best_move` rule.
///
/// ### Params
///
/// * `indptr` - Row offsets
/// * `indices` - Column indices
/// * `weights` - Edge weights
/// * `strength` - Node weight
/// * `kint` - Fixed-point node weight
/// * `comm` - Cluster per node
/// * `tot` - Fixed-point cluster totals
/// * `size` - Cluster sizes
/// * `scale` - Resolution over `2m`, divided by the fixed-point scale
/// * `alist` - Pending lists of all classes
/// * `goff` - Offset of this class in `alist`
/// * `cur` - Pending count of this class
/// * `affected` - Pending flag per node; cleared here
/// * `out_to` - Target cluster, [`STAY`] or [`NEW`], per pending slot
/// * `out_wto` - Edge weight to the target
/// * `out_wfrom` - Edge weight to the own cluster, self excluded
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn lm_decide(
    indptr: &Tensor<u32>,
    indices: &Tensor<u32>,
    weights: &Tensor<f32>,
    strength: &Tensor<f32>,
    kint: &Tensor<u32>,
    comm: &Tensor<u32>,
    tot: &Tensor<u32>,
    size: &Tensor<u32>,
    scale: f32,
    alist: &Tensor<u32>,
    goff: u32,
    cur: &Tensor<u32>,
    affected: &mut Tensor<u32>,
    out_to: &mut Tensor<u32>,
    out_wto: &mut Tensor<f32>,
    out_wfrom: &mut Tensor<f32>,
) {
    let count = cur[0] as usize;
    let lane = UNIT_POS_PLANE;
    let pd = PLANE_DIM as usize;
    // 1D workgroup: WGSL rejects subgroup builtins in a multi-dimensional one
    let planes = (CUBE_DIM / PLANE_DIM) as usize;
    let stride = CUBE_COUNT * planes;
    let mut i = CUBE_POS * planes + (UNIT_POS / PLANE_DIM) as usize;
    while i < count {
        let v = alist[goff as usize + i] as usize;
        if lane == 0 {
            affected[v] = 0u32;
        }
        let start = indptr[v] as usize;
        let end = indptr[v + 1] as usize;
        let own = comm[v];
        let k = strength[v];
        let chunks = (end - start + pd - 1) / pd;

        let mut w_from = 0.0f32;
        let mut best_c = u32::new(LIST_END as i64);
        let mut best_g = 0.0f32;
        let mut best_w = 0.0f32;
        for t in 0..chunks {
            let ei = t * pd + lane as usize;
            let mut c = 0u32;
            let mut ok = 0u32;
            if start + ei < end {
                let u = indices[start + ei];
                if u != v as u32 {
                    c = comm[u as usize];
                    ok = 1u32;
                }
            }
            // this lane's cluster summed over the row, and whether its edge is
            // the cluster's first occurrence
            let mut acc = 0.0f32;
            let mut first = ok;
            for t2 in 0..chunks {
                let e2 = t2 * pd + lane as usize;
                let mut c2 = 0u32;
                let mut w2 = 0.0f32;
                let mut ok2 = 0u32;
                if start + e2 < end {
                    let u2 = indices[start + e2];
                    if u2 != v as u32 {
                        c2 = comm[u2 as usize];
                        w2 = weights[start + e2];
                        ok2 = 1u32;
                    }
                }
                for j in 0..pd {
                    let cj = plane_shuffle(c2, j as u32);
                    let wj = plane_shuffle(w2, j as u32);
                    let okj = plane_shuffle(ok2, j as u32);
                    // nested: cubecl does not promise short-circuit `&&`
                    if okj == 1u32 {
                        if cj == c {
                            acc += wj;
                            if t2 * pd + j < ei {
                                first = 0u32;
                            }
                        }
                    }
                }
            }

            let mut wf = 0.0f32;
            let mut g = f32::min_value();
            let mut cand = u32::new(LIST_END as i64);
            if first == 1u32 {
                if c == own {
                    wf = acc;
                } else {
                    g = acc - k * f32::cast_from(tot[c as usize]) * scale;
                    cand = c;
                }
            }
            w_from += plane_sum(wf);

            let gm = plane_max(g);
            let mut cid = u32::new(LIST_END as i64);
            if cand != LIST_END {
                if g == gm {
                    cid = cand;
                }
            }
            let cm = plane_min(cid);
            if cm != LIST_END {
                // a cluster has one first occurrence, so one lane contributes
                let mut ws = 0.0f32;
                if cand == cm {
                    ws = acc;
                }
                let wm = plane_sum(ws);
                let mut take = best_c == LIST_END;
                if gm > best_g {
                    take = true;
                }
                if gm == best_g {
                    if cm < best_c {
                        take = true;
                    }
                }
                if take {
                    best_c = cm;
                    best_g = gm;
                    best_w = wm;
                }
            }
        }

        let stay = w_from - k * f32::cast_from(tot[own as usize] - kint[v]) * scale;
        let mut to = u32::new(STAY as i64);
        let mut wto = 0.0f32;
        if best_c != LIST_END {
            if best_g > stay {
                to = best_c;
                wto = best_w;
            }
        }
        if to == STAY {
            if size[own as usize] > 1u32 {
                if stay < 0.0f32 {
                    to = NEW;
                }
            }
        }
        if lane == 0 {
            out_to[i] = to;
            out_wto[i] = wto;
            out_wfrom[i] = w_from;
        }
        i += stride;
    }
}

/// Source and target cluster of each proposal (a new cluster is `n + v`), and
/// reset the list heads they use.
///
/// ### Params
///
/// * `alist` - Pending lists
/// * `goff` - Offset of this class in `alist`
/// * `cur` - Pending count of this class
/// * `n` - Number of nodes
/// * `comm` - Cluster per node
/// * `out_to` - Decided targets
/// * `r_from` - Output, source cluster
/// * `r_to` - Output, target cluster
/// * `heads_to` - List head per target cluster
/// * `heads_from` - List head per source cluster
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn lm_resolve(
    alist: &Tensor<u32>,
    goff: u32,
    cur: &Tensor<u32>,
    n: u32,
    comm: &Tensor<u32>,
    out_to: &Tensor<u32>,
    r_from: &mut Tensor<u32>,
    r_to: &mut Tensor<u32>,
    heads_to: &mut Tensor<u32>,
    heads_from: &mut Tensor<u32>,
) {
    let count = cur[0] as usize;
    let stride = CUBE_COUNT * CUBE_DIM as usize;
    let mut i = ABSOLUTE_POS;
    while i < count {
        let to0 = out_to[i];
        if to0 != STAY {
            let v = alist[goff as usize + i];
            let from = comm[v as usize];
            let mut to = to0;
            if to0 == NEW {
                to = n + v;
            }
            r_from[i] = from;
            r_to[i] = to;
            heads_to[to as usize] = LIST_END;
            heads_from[from as usize] = LIST_END;
        }
        i += stride;
    }
}

/// Link each proposal into its target's and its source's list. Lists are
/// unordered; only their sets matter.
///
/// ### Params
///
/// * `cur` - Pending count of this class
/// * `out_to` - Decided targets
/// * `r_from` - Source clusters
/// * `r_to` - Target clusters
/// * `heads_to` - List head per target cluster
/// * `heads_from` - List head per source cluster
/// * `next_to` - Output, next proposal into the same target
/// * `next_from` - Output, next proposal out of the same source
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn lm_link(
    cur: &Tensor<u32>,
    out_to: &Tensor<u32>,
    r_from: &Tensor<u32>,
    r_to: &Tensor<u32>,
    heads_to: &mut Tensor<Atomic<u32>>,
    heads_from: &mut Tensor<Atomic<u32>>,
    next_to: &mut Tensor<u32>,
    next_from: &mut Tensor<u32>,
) {
    let count = cur[0] as usize;
    let stride = CUBE_COUNT * CUBE_DIM as usize;
    let mut i = ABSOLUTE_POS;
    while i < count {
        if out_to[i] != STAY {
            next_to[i] = heads_to[r_to[i] as usize].exchange(i as u32);
            next_from[i] = heads_from[r_from[i] as usize].exchange(i as u32);
        }
        i += stride;
    }
}

/// OrderFree acceptance: re-check each proposal against the frozen totals with
/// every proposal of a lower node id counted into its target and out of its
/// source.
///
/// ### Params
///
/// * `alist` - Pending lists
/// * `goff` - Offset of this class in `alist`
/// * `cur` - Pending count of this class
/// * `strength` - Node weight
/// * `kint` - Fixed-point node weight
/// * `tot` - Fixed-point cluster totals
/// * `size` - Cluster sizes
/// * `scale` - Resolution over `2m`, divided by the fixed-point scale
/// * `out_to` - Decided targets
/// * `out_wto` - Edge weight to the target
/// * `out_wfrom` - Edge weight to the own cluster
/// * `r_from` - Source clusters
/// * `r_to` - Target clusters
/// * `heads_to` - List head per target cluster
/// * `heads_from` - List head per source cluster
/// * `next_to` - Next proposal into the same target
/// * `next_from` - Next proposal out of the same source
/// * `acc` - Output, 1 if accepted
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn lm_accept(
    alist: &Tensor<u32>,
    goff: u32,
    cur: &Tensor<u32>,
    strength: &Tensor<f32>,
    kint: &Tensor<u32>,
    tot: &Tensor<u32>,
    size: &Tensor<u32>,
    scale: f32,
    out_to: &Tensor<u32>,
    out_wto: &Tensor<f32>,
    out_wfrom: &Tensor<f32>,
    r_from: &Tensor<u32>,
    r_to: &Tensor<u32>,
    heads_to: &Tensor<u32>,
    heads_from: &Tensor<u32>,
    next_to: &Tensor<u32>,
    next_from: &Tensor<u32>,
    acc: &mut Tensor<u32>,
) {
    let count = cur[0] as usize;
    let stride = CUBE_COUNT * CUBE_DIM as usize;
    let mut i = ABSOLUTE_POS;
    while i < count {
        let mut ok = 0u32;
        let to0 = out_to[i];
        if to0 != STAY {
            let v = alist[goff as usize + i];
            let from = r_from[i] as usize;
            let to = r_to[i] as usize;
            let mut inc = 0u32;
            let mut j = heads_to[to];
            while j != LIST_END {
                let u = alist[goff as usize + j as usize];
                if u < v {
                    inc += kint[u as usize];
                }
                j = next_to[j as usize];
            }
            let mut outg = 0u32;
            let mut outc = 0u32;
            j = heads_from[from];
            while j != LIST_END {
                let u = alist[goff as usize + j as usize];
                if u < v {
                    outg += kint[u as usize];
                    outc += 1u32;
                }
                j = next_from[j as usize];
            }
            let k = strength[v as usize];
            let kv = kint[v as usize];
            let stay = out_wfrom[i] - k * f32::cast_from(tot[from] - kv - outg) * scale;
            if to0 == NEW {
                // the new id must be empty and v must not be left alone already
                if size[to] == 0u32 {
                    if size[from] - outc > 1u32 {
                        if stay < 0.0f32 {
                            ok = 1u32;
                        }
                    }
                }
            } else {
                let go = out_wto[i] - k * f32::cast_from(tot[to] + inc) * scale;
                if go > stay {
                    ok = 1u32;
                }
            }
        }
        acc[i] = ok;
        i += stride;
    }
}

/// Apply the accepted moves: integer atomics on totals and sizes, new labels,
/// a move count.
///
/// ### Params
///
/// * `alist` - Pending lists
/// * `goff` - Offset of this class in `alist`
/// * `cur` - Pending count of this class
/// * `kint` - Fixed-point node weight
/// * `acc` - 1 if accepted
/// * `r_from` - Source clusters
/// * `r_to` - Target clusters
/// * `tot` - Fixed-point cluster totals
/// * `size` - Cluster sizes
/// * `comm` - Cluster per node
/// * `moves` - Move counter of this round
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn lm_apply(
    alist: &Tensor<u32>,
    goff: u32,
    cur: &Tensor<u32>,
    kint: &Tensor<u32>,
    acc: &Tensor<u32>,
    r_from: &Tensor<u32>,
    r_to: &Tensor<u32>,
    tot: &mut Tensor<Atomic<u32>>,
    size: &mut Tensor<Atomic<u32>>,
    comm: &mut Tensor<u32>,
    moves: &mut Tensor<Atomic<u32>>,
) {
    let count = cur[0] as usize;
    let stride = CUBE_COUNT * CUBE_DIM as usize;
    let mut i = ABSOLUTE_POS;
    while i < count {
        if acc[i] == 1u32 {
            let v = alist[goff as usize + i] as usize;
            let kv = kint[v];
            let from = r_from[i] as usize;
            let to = r_to[i] as usize;
            tot[from].fetch_sub(kv);
            tot[to].fetch_add(kv);
            size[from].fetch_sub(1u32);
            size[to].fetch_add(1u32);
            comm[v] = r_to[i];
            moves[0].fetch_add(1u32);
        }
        i += stride;
    }
}

/// Flag the neighbours of each moved node that now sit in another cluster and
/// append them to their class's pending list. A class is an independent set,
/// so the appends never land in this class's region.
///
/// ### Params
///
/// * `indptr` - Row offsets
/// * `indices` - Column indices
/// * `comm` - Cluster per node, after the moves
/// * `alist` - Pending lists
/// * `goff` - Offset of this class in `alist`
/// * `cur` - Pending count of this class
/// * `acc` - 1 if accepted
/// * `affected` - Pending flag per node
/// * `colour` - Class per node
/// * `goffs` - Offset of each class in `alist`
/// * `acount` - Pending count per class
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn lm_mark(
    indptr: &Tensor<u32>,
    indices: &Tensor<u32>,
    comm: &Tensor<u32>,
    alist: &mut Tensor<u32>,
    goff: u32,
    cur: &Tensor<u32>,
    acc: &Tensor<u32>,
    affected: &mut Tensor<Atomic<u32>>,
    colour: &Tensor<u32>,
    goffs: &Tensor<u32>,
    acount: &mut Tensor<Atomic<u32>>,
) {
    let count = cur[0] as usize;
    let stride = CUBE_COUNT * CUBE_DIM as usize;
    let mut i = ABSOLUTE_POS;
    while i < count {
        if acc[i] == 1u32 {
            let v = alist[goff as usize + i] as usize;
            let c = comm[v];
            for e in indptr[v] as usize..indptr[v + 1] as usize {
                let u = indices[e] as usize;
                if comm[u] != c {
                    // plain load first: most neighbours are already flagged
                    if affected[u].load() == 0u32 {
                        if affected[u].exchange(1u32) == 0u32 {
                            let gu = colour[u] as usize;
                            let slot = acount[gu].fetch_add(1u32);
                            alist[(goffs[gu] + slot) as usize] = u as u32;
                        }
                    }
                }
            }
        }
        i += stride;
    }
}

//////////
// Host //
//////////

/// Level-0 graph resident on the device.
struct DevGraph {
    /// Number of nodes
    n: usize,
    /// Row offsets, length n + 1
    indptr: GpuTensor<u32>,
    /// Column indices
    indices: GpuTensor<u32>,
    /// Edge weights
    weights: GpuTensor<f32>,
    /// Node weights
    strength: GpuTensor<f32>,
    /// Fixed-point node weights
    kint: GpuTensor<u32>,
    /// Fixed-point scale, a power of two
    fx_scale: f64,
}

/// [`LocalMover`] that runs coloured local moving on a cubecl device.
///
/// Needs a device with plane operations of one fixed width; see
/// [`gpu_supported`].
pub struct LeidenGpuMover<'a> {
    /// Compute client
    client: &'a Client,
    /// The prepared graph
    graph: Option<DevGraph>,
}

impl<'a> LeidenGpuMover<'a> {
    /// Create a mover on `client`.
    ///
    /// ### Params
    ///
    /// * `client` - CubeCL compute client
    ///
    /// ### Returns
    ///
    /// The mover, with no graph yet
    pub fn new(client: &'a Client) -> Self {
        Self {
            client,
            graph: None,
        }
    }
}

/// Grid for a thread-per-entry launch over at most `len` entries.
///
/// ### Params
///
/// * `len` - Upper bound on the entries
///
/// ### Returns
///
/// `(cube count, cube dim)`, capped at [`GRID_CAP`] threads
fn thread_grid(len: usize) -> (CubeCount, CubeDim) {
    let threads = len.clamp(1, GRID_CAP) as u32;
    (
        CubeCount::Static(threads.div_ceil(BLOCK), 1, 1),
        CubeDim::new_1d(BLOCK),
    )
}

/// Grid for a plane-per-entry launch over at most `len` entries.
///
/// ### Params
///
/// * `len` - Upper bound on the entries
/// * `plane` - Plane width of the device
///
/// ### Returns
///
/// `(cube count, cube dim)`, capped at [`GRID_CAP`] planes
fn plane_grid(len: usize, plane: u32) -> (CubeCount, CubeDim) {
    let planes = len.clamp(1, GRID_CAP) as u32;
    (
        CubeCount::Static(planes.div_ceil(PLANES_PER_CUBE), 1, 1),
        CubeDim::new_1d(plane * PLANES_PER_CUBE),
    )
}

/// Plane width of the device, if the GPU path can run on it.
///
/// ### Params
///
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// The plane width, or `None` without plane operations or with a plane width
/// the device only reports as a range
pub fn gpu_supported(client: &Client) -> Option<u32> {
    let limits = GpuLimits::from_client(client);
    let ops = client
        .properties()
        .features
        .plane
        .contains(cubecl::features::Plane::Ops);
    (ops && limits.plane_size_min == limits.plane_size_max && limits.plane_size_min > 0)
        .then_some(limits.plane_size_min)
}

impl LocalMover for LeidenGpuMover<'_> {
    type Error = GpuErrors;

    fn prepare<W: ToPrimitive + Sync>(
        &mut self,
        indptr: &[usize],
        indices: &[u32],
        weights: &[W],
        node_w: &[f64],
    ) -> Result<(), GpuErrors> {
        let n = node_w.len();
        let two_m: f64 = node_w.iter().sum();
        let fx_scale = 2f64.powi(FX_BITS - two_m.log2().ceil() as i32);
        let kint: Vec<u32> = node_w
            .iter()
            .map(|&k| (k * fx_scale).round() as u32)
            .collect();
        let off: Vec<u32> = indptr.iter().map(|&o| o as u32).collect();
        let w: Vec<f32> = weights
            .iter()
            .map(|w| w.to_f32().expect("f32 weight"))
            .collect();
        let s: Vec<f32> = node_w.iter().map(|&k| k as f32).collect();
        let c = self.client;
        self.graph = Some(DevGraph {
            n,
            indptr: GpuTensor::from_slice(&off, vec![n + 1], c)?,
            indices: GpuTensor::from_slice(indices, vec![indices.len()], c)?,
            weights: GpuTensor::from_slice(&w, vec![w.len()], c)?,
            strength: GpuTensor::from_slice(&s, vec![n], c)?,
            kint: GpuTensor::from_slice(&kint, vec![n], c)?,
            fx_scale,
        });
        Ok(())
    }

    fn move_nodes(
        &mut self,
        membership: &mut [u32],
        classes: &[Vec<u32>],
        colour: &[u32],
        res: f64,
    ) -> Result<bool, GpuErrors> {
        let dev = self.run(membership, classes, colour, res)?;
        // new clusters carry ids n + v; fold every id back below n
        let labels = dev.comm.read(self.client)?;
        let mut map = vec![u32::MAX; labels.len() * 2];
        let mut next = 0u32;
        for (m, &l) in membership.iter_mut().zip(&labels) {
            let slot = &mut map[l as usize];
            if *slot == u32::MAX {
                *slot = next;
                next += 1;
            }
            *m = *slot;
        }
        Ok(dev.changed)
    }
}

/// Device state left by [`LeidenGpuMover::run`].
struct DevPartition {
    /// Cluster per node, ids `< 2n`
    comm: GpuTensor<u32>,
    /// Fixed-point cluster totals, length 2n; read by the tests only
    #[cfg(test)]
    tot: GpuTensor<u32>,
    /// Cluster sizes, length 2n; read by the tests only
    #[cfg(test)]
    size: GpuTensor<u32>,
    /// Whether any node moved
    changed: bool,
}

impl LeidenGpuMover<'_> {
    /// Coloured local moving on the device, see [`LocalMover::move_nodes`].
    ///
    /// ### Params
    ///
    /// * `membership` - Initial partition, labels `< n`
    /// * `classes` - Colour classes
    /// * `colour` - Colour per node
    /// * `res` - Resolution divided by `2m`
    ///
    /// ### Returns
    ///
    /// The partition, totals and sizes, still on the device
    fn run(
        &self,
        membership: &[u32],
        classes: &[Vec<u32>],
        colour: &[u32],
        res: f64,
    ) -> Result<DevPartition, GpuErrors> {
        let d = self.graph.as_ref().expect("prepare runs first");
        let c = self.client;
        let n = d.n;
        let plane = gpu_supported(c).expect("checked by the caller");
        let scale = (res / d.fx_scale) as f32;

        let lens: Vec<u32> = classes.iter().map(|x| x.len() as u32).collect();
        let mut goffs = vec![0u32; classes.len() + 1];
        for (g, &l) in lens.iter().enumerate() {
            goffs[g + 1] = goffs[g] + l;
        }
        let alist: Vec<u32> = classes.concat();
        let max_len = lens.iter().copied().max().unwrap_or(1).max(1) as usize;

        let comm = GpuTensor::from_slice(membership, vec![n], c)?;
        let tot = GpuTensor::from_slice(&vec![0u32; 2 * n], vec![2 * n], c)?;
        let size = GpuTensor::from_slice(&vec![0u32; 2 * n], vec![2 * n], c)?;
        let affected = GpuTensor::from_slice(&vec![1u32; n], vec![n], c)?;
        let alist = GpuTensor::from_slice(&alist, vec![n], c)?;
        let colour = GpuTensor::from_slice(colour, vec![n], c)?;
        let goffs_d = GpuTensor::from_slice(&goffs, vec![goffs.len()], c)?;
        let acount = GpuTensor::from_slice(&lens, vec![lens.len()], c)?;
        let heads_to = GpuTensor::<u32>::empty(vec![2 * n], c)?;
        let heads_from = GpuTensor::<u32>::empty(vec![2 * n], c)?;
        let cur = GpuTensor::<u32>::empty(vec![1], c)?;
        let moves = GpuTensor::<u32>::empty(vec![1], c)?;
        let out_to = GpuTensor::<u32>::empty(vec![max_len], c)?;
        let out_wto = GpuTensor::<f32>::empty(vec![max_len], c)?;
        let out_wfrom = GpuTensor::<f32>::empty(vec![max_len], c)?;
        let r_from = GpuTensor::<u32>::empty(vec![max_len], c)?;
        let r_to = GpuTensor::<u32>::empty(vec![max_len], c)?;
        let next_to = GpuTensor::<u32>::empty(vec![max_len], c)?;
        let next_from = GpuTensor::<u32>::empty(vec![max_len], c)?;
        let acc = GpuTensor::<u32>::empty(vec![max_len], c)?;

        let one = (CubeCount::Static(1, 1, 1), CubeDim::new_1d(1));
        unsafe {
            let (cc, cd) = (
                CubeCount::Static((n as u32).div_ceil(BLOCK), 1, 1),
                CubeDim::new_1d(BLOCK),
            );
            lm_init::launch_unchecked(
                c,
                cc,
                cd,
                comm.into_tensor_arg(),
                d.kint.into_tensor_arg(),
                tot.into_tensor_arg(),
                size.into_tensor_arg(),
                n as u32,
            );
        }

        let mut changed = false;
        for _ in 0..MAX_ROUNDS {
            unsafe {
                lm_zero::launch_unchecked(c, one.0.clone(), one.1, moves.into_tensor_arg());
            }
            for (g, &len) in lens.iter().enumerate() {
                let len = len as usize;
                let goff = goffs[g];
                let (tc, td) = thread_grid(len);
                let (pc, pdim) = plane_grid(len, plane);
                unsafe {
                    lm_begin::launch_unchecked(
                        c,
                        one.0.clone(),
                        one.1,
                        acount.into_tensor_arg(),
                        g as u32,
                        cur.into_tensor_arg(),
                    );
                    lm_decide::launch_unchecked(
                        c,
                        pc,
                        pdim,
                        d.indptr.into_tensor_arg(),
                        d.indices.into_tensor_arg(),
                        d.weights.into_tensor_arg(),
                        d.strength.into_tensor_arg(),
                        d.kint.into_tensor_arg(),
                        comm.into_tensor_arg(),
                        tot.into_tensor_arg(),
                        size.into_tensor_arg(),
                        scale,
                        alist.into_tensor_arg(),
                        goff,
                        cur.into_tensor_arg(),
                        affected.into_tensor_arg(),
                        out_to.into_tensor_arg(),
                        out_wto.into_tensor_arg(),
                        out_wfrom.into_tensor_arg(),
                    );
                    lm_resolve::launch_unchecked(
                        c,
                        tc.clone(),
                        td,
                        alist.into_tensor_arg(),
                        goff,
                        cur.into_tensor_arg(),
                        n as u32,
                        comm.into_tensor_arg(),
                        out_to.into_tensor_arg(),
                        r_from.into_tensor_arg(),
                        r_to.into_tensor_arg(),
                        heads_to.into_tensor_arg(),
                        heads_from.into_tensor_arg(),
                    );
                    lm_link::launch_unchecked(
                        c,
                        tc.clone(),
                        td,
                        cur.into_tensor_arg(),
                        out_to.into_tensor_arg(),
                        r_from.into_tensor_arg(),
                        r_to.into_tensor_arg(),
                        heads_to.into_tensor_arg(),
                        heads_from.into_tensor_arg(),
                        next_to.into_tensor_arg(),
                        next_from.into_tensor_arg(),
                    );
                    lm_accept::launch_unchecked(
                        c,
                        tc.clone(),
                        td,
                        alist.into_tensor_arg(),
                        goff,
                        cur.into_tensor_arg(),
                        d.strength.into_tensor_arg(),
                        d.kint.into_tensor_arg(),
                        tot.into_tensor_arg(),
                        size.into_tensor_arg(),
                        scale,
                        out_to.into_tensor_arg(),
                        out_wto.into_tensor_arg(),
                        out_wfrom.into_tensor_arg(),
                        r_from.into_tensor_arg(),
                        r_to.into_tensor_arg(),
                        heads_to.into_tensor_arg(),
                        heads_from.into_tensor_arg(),
                        next_to.into_tensor_arg(),
                        next_from.into_tensor_arg(),
                        acc.into_tensor_arg(),
                    );
                    lm_apply::launch_unchecked(
                        c,
                        tc.clone(),
                        td,
                        alist.into_tensor_arg(),
                        goff,
                        cur.into_tensor_arg(),
                        d.kint.into_tensor_arg(),
                        acc.into_tensor_arg(),
                        r_from.into_tensor_arg(),
                        r_to.into_tensor_arg(),
                        tot.into_tensor_arg(),
                        size.into_tensor_arg(),
                        comm.into_tensor_arg(),
                        moves.into_tensor_arg(),
                    );
                    lm_mark::launch_unchecked(
                        c,
                        tc,
                        td,
                        d.indptr.into_tensor_arg(),
                        d.indices.into_tensor_arg(),
                        comm.into_tensor_arg(),
                        alist.into_tensor_arg(),
                        goff,
                        cur.into_tensor_arg(),
                        acc.into_tensor_arg(),
                        affected.into_tensor_arg(),
                        colour.into_tensor_arg(),
                        goffs_d.into_tensor_arg(),
                        acount.into_tensor_arg(),
                    );
                }
            }
            let moved = moves.clone().read(c)?[0];
            if moved == 0 {
                break;
            }
            changed = true;
        }

        Ok(DevPartition {
            comm,
            #[cfg(test)]
            tot,
            #[cfg(test)]
            size,
            changed,
        })
    }
}

/// Leiden or Louvain with level-0 local moving on the GPU.
///
/// Falls back to the CPU engine with [`LocalMoving::Coloured`] when the graph
/// has fewer than [`LEIDEN_GPU_MIN_NODES`] nodes, a row longer than
/// [`PLANE_DEG`], or the device lacks fixed-width plane operations.
///
/// ### Params
///
/// * `graph` - Undirected [`SparseGraph`]
/// * `params` - The [`LeidenParams`]; `local_moving` applies to the coarser
///   levels and to the fallback
/// * `max_evals` - Node-evaluation cap per CPU local-moving phase, in units
///   of n
/// * `multi_level` - `false` stops after the first local-moving phase
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// Community per node, `0` the largest
pub fn community_detection_gpu<T>(
    graph: &SparseGraph<T>,
    params: &LeidenParams,
    max_evals: usize,
    multi_level: bool,
    client: &Client,
) -> Result<Vec<usize>, GpuErrors>
where
    T: BixverseFloat + Send + Sync + std::iter::Sum,
{
    let n = graph.get_node_number();
    let max_deg = (0..n)
        .map(|i| graph.get_neighbours_raw(i).0.len())
        .max()
        .unwrap_or(0);
    if n < LEIDEN_GPU_MIN_NODES || max_deg > PLANE_DEG || gpu_supported(client).is_none() {
        let cpu = LeidenParams {
            local_moving: LocalMoving::Coloured,
            ..*params
        };
        return Ok(community_detection(graph, &cpu, max_evals, multi_level)?);
    }
    let mut mover = LeidenGpuMover::new(client);
    community_detection_with(graph, params, max_evals, multi_level, &mut mover)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use bixverse_core::core::math::sparse::coo_to_csr;
    use bixverse_methods::graph::leiden::modularity;
    use cubecl::wgpu::WgpuDevice;
    use rand::prelude::*;

    /// A client on the default device, or `None` when the machine has no
    /// usable GPU or the device lacks fixed-width planes; tests then return
    /// early.
    fn try_client() -> Option<Client> {
        let device = WgpuDevice::default();
        let client = std::panic::catch_unwind(|| cubecl::Device::from(device).client()).ok()?;
        gpu_supported(&client).map(|_| client)
    }

    /// Undirected graph from an edge list, both directions stored.
    fn build(n: usize, edges: &[(usize, usize, f64)]) -> SparseGraph<f64> {
        let mut rows = Vec::with_capacity(edges.len() * 2);
        let mut cols = Vec::with_capacity(edges.len() * 2);
        let mut vals = Vec::with_capacity(edges.len() * 2);
        for &(u, v, w) in edges {
            rows.extend([u, v]);
            cols.extend([v, u]);
            vals.extend([w, w]);
        }
        let csr = coo_to_csr(&rows.index_cast(), &cols.index_cast(), &vals, (n, n));
        SparseGraph::new(n, csr, false)
    }

    /// Planted partition: `k` blocks of `size` nodes, `d_in` random edges per
    /// node inside its block and `d_out` outside, integer weights 1..=3.
    fn planted(k: usize, size: usize, d_in: usize, d_out: usize, seed: u64) -> SparseGraph<f64> {
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
        build(n, &edges)
    }

    /// Sequential greedy colouring in node order.
    fn colouring(g: &SparseGraph<f64>) -> (Vec<Vec<u32>>, Vec<u32>) {
        let n = g.get_node_number();
        let mut colour = vec![u32::MAX; n];
        let mut classes: Vec<Vec<u32>> = Vec::new();
        for v in 0..n {
            let used: Vec<u32> = g
                .get_neighbours_raw(v)
                .0
                .iter()
                .map(|&u| colour[u as usize])
                .collect();
            let c = (0..).find(|c| !used.contains(c)).unwrap();
            if c as usize == classes.len() {
                classes.push(Vec::new());
            }
            colour[v] = c;
            classes[c as usize].push(v as u32);
        }
        (classes, colour)
    }

    /// CSR arrays and node weights of `g`.
    fn csr(g: &SparseGraph<f64>) -> (Vec<usize>, Vec<u32>, Vec<f64>, Vec<f64>) {
        let n = g.get_node_number();
        let (mut indptr, mut indices, mut weights) = (vec![0usize], Vec::new(), Vec::new());
        for v in 0..n {
            let (t, w) = g.get_neighbours_raw(v);
            indices.extend_from_slice(t);
            weights.extend_from_slice(w);
            indptr.push(indices.len());
        }
        let node_w = (0..n)
            .map(|v| g.get_neighbours_raw(v).1.iter().sum())
            .collect();
        (indptr, indices, weights, node_w)
    }

    fn params(refine: bool, seed: u64) -> LeidenParams {
        LeidenParams::new(1.0, 0.01, 2, refine, seed)
    }

    fn run_gpu(g: &SparseGraph<f64>, refine: bool, seed: u64, client: &Client) -> Vec<usize> {
        let mut mover = LeidenGpuMover::new(client);
        community_detection_with(g, &params(refine, seed), usize::MAX, true, &mut mover).unwrap()
    }

    fn run_cpu(g: &SparseGraph<f64>, refine: bool, seed: u64) -> Vec<usize> {
        let p = LeidenParams {
            local_moving: LocalMoving::Coloured,
            ..params(refine, seed)
        };
        community_detection(g, &p, usize::MAX, true).unwrap()
    }

    /// After one local-moving phase the device totals and sizes equal what
    /// the host recomputes from the labels, exactly, and the run moved nodes.
    /// Degree up to ~80 covers rows of several plane chunks.
    #[test]
    fn test_leiden_gpu_totals_match_labels() {
        let Some(client) = try_client() else { return };
        let g = planted(6, 300, 40, 2, 3);
        let (indptr, indices, weights, node_w) = csr(&g);
        let n = node_w.len();
        let mut mover = LeidenGpuMover::new(&client);
        mover.prepare(&indptr, &indices, &weights, &node_w).unwrap();
        let (classes, colour) = colouring(&g);
        let membership: Vec<u32> = (0..n as u32).collect();
        let two_m: f64 = node_w.iter().sum();
        let dev = mover
            .run(&membership, &classes, &colour, 1.0 / two_m)
            .unwrap();
        assert!(dev.changed);

        let comm = dev.comm.read(&client).unwrap();
        let tot = dev.tot.read(&client).unwrap();
        let size = dev.size.read(&client).unwrap();
        let fx = mover.graph.as_ref().unwrap().fx_scale;
        let (mut want_tot, mut want_size) = (vec![0u32; 2 * n], vec![0u32; 2 * n]);
        for v in 0..n {
            want_tot[comm[v] as usize] += (node_w[v] * fx).round() as u32;
            want_size[comm[v] as usize] += 1;
        }
        assert_eq!(tot, want_tot);
        assert_eq!(size, want_size);
        assert!(want_size.iter().filter(|&&s| s > 0).count() < n / 10);
    }

    /// Same seed, same labels; for Leiden and Louvain.
    #[test]
    fn test_leiden_gpu_reproducible() {
        let Some(client) = try_client() else { return };
        let g = planted(20, 400, 8, 3, 11);
        for refine in [true, false] {
            let a = run_gpu(&g, refine, 5, &client);
            let b = run_gpu(&g, refine, 5, &client);
            assert_eq!(a, b, "refine {refine}");
        }
    }

    /// Clean planted blocks are recovered exactly.
    #[test]
    fn test_leiden_gpu_recovers_planted_blocks() {
        let Some(client) = try_client() else { return };
        let g = planted(8, 600, 10, 1, 7);
        for refine in [true, false] {
            let labels = run_gpu(&g, refine, 1, &client);
            assert_eq!(labels.iter().max().unwrap() + 1, 8, "refine {refine}");
            for v in 0..4800 {
                assert_eq!(labels[v], labels[(v / 600) * 600], "refine {refine}");
            }
        }
    }

    /// On a noisy planted partition and on karate the GPU lands within 1% of
    /// the CPU Coloured modularity, for Leiden and Louvain.
    #[test]
    fn test_leiden_gpu_modularity_close_to_cpu() {
        let Some(client) = try_client() else { return };
        let karate: [(usize, usize); 78] = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 17),
            (0, 19),
            (0, 21),
            (0, 31),
            (1, 2),
            (1, 3),
            (1, 7),
            (1, 13),
            (1, 17),
            (1, 19),
            (1, 21),
            (1, 30),
            (2, 3),
            (2, 7),
            (2, 27),
            (2, 28),
            (2, 32),
            (2, 9),
            (2, 8),
            (2, 13),
            (3, 7),
            (3, 12),
            (3, 13),
            (4, 6),
            (4, 10),
            (5, 6),
            (5, 10),
            (5, 16),
            (6, 16),
            (8, 30),
            (8, 32),
            (8, 33),
            (9, 33),
            (13, 33),
            (14, 32),
            (14, 33),
            (15, 32),
            (15, 33),
            (18, 32),
            (18, 33),
            (19, 33),
            (20, 32),
            (20, 33),
            (22, 32),
            (22, 33),
            (23, 25),
            (23, 27),
            (23, 32),
            (23, 33),
            (23, 29),
            (24, 25),
            (24, 27),
            (24, 31),
            (25, 31),
            (26, 29),
            (26, 33),
            (27, 33),
            (28, 31),
            (28, 33),
            (29, 32),
            (29, 33),
            (30, 32),
            (30, 33),
            (31, 32),
            (31, 33),
            (32, 33),
        ];
        let karate: Vec<(usize, usize, f64)> = karate.iter().map(|&(u, v)| (u, v, 1.0)).collect();
        for g in [planted(20, 400, 8, 3, 11), build(34, &karate)] {
            for refine in [true, false] {
                let q_cpu = modularity(&g, &run_cpu(&g, refine, 2), 1.0);
                let q_gpu = modularity(&g, &run_gpu(&g, refine, 2, &client), 1.0);
                assert!(
                    q_gpu > q_cpu - 0.01 * q_cpu.abs(),
                    "refine {refine}: gpu {q_gpu} cpu {q_cpu}"
                );
            }
        }
    }

    /// [`LeidenGpuMover`] with wall-clock time per call.
    #[cfg(feature = "large-test")]
    struct TimedMover<'a> {
        /// The mover
        inner: LeidenGpuMover<'a>,
        /// Seconds in `prepare`
        prepare: f64,
        /// Seconds in `move_nodes`
        moving: f64,
    }

    #[cfg(feature = "large-test")]
    impl LocalMover for TimedMover<'_> {
        type Error = GpuErrors;

        fn prepare<W: ToPrimitive + Sync>(
            &mut self,
            indptr: &[usize],
            indices: &[u32],
            weights: &[W],
            node_w: &[f64],
        ) -> Result<(), GpuErrors> {
            let t = std::time::Instant::now();
            self.inner.prepare(indptr, indices, weights, node_w)?;
            self.prepare += t.elapsed().as_secs_f64();
            Ok(())
        }

        fn move_nodes(
            &mut self,
            membership: &mut [u32],
            classes: &[Vec<u32>],
            colour: &[u32],
            res: f64,
        ) -> Result<bool, GpuErrors> {
            let t = std::time::Instant::now();
            let changed = self.inner.move_nodes(membership, classes, colour, res)?;
            self.moving += t.elapsed().as_secs_f64();
            Ok(changed)
        }
    }

    /// Edge list written by R: `n`, `m` as i32, then `m` sources, `m` targets
    /// (0-based i32) and `m` f32 weights.
    #[cfg(feature = "large-test")]
    fn read_edge_list(path: &str) -> SparseGraph<f64> {
        let bytes = std::fs::read(path).expect("graph file");
        let word = |i: usize| <[u8; 4]>::try_from(&bytes[4 * i..4 * i + 4]).unwrap();
        let (n, m) = (
            i32::from_le_bytes(word(0)) as usize,
            i32::from_le_bytes(word(1)) as usize,
        );
        let edges: Vec<(usize, usize, f64)> = (0..m)
            .map(|e| {
                (
                    i32::from_le_bytes(word(2 + e)) as usize,
                    i32::from_le_bytes(word(2 + m + e)) as usize,
                    f32::from_le_bytes(word(2 + 2 * m + e)) as f64,
                )
            })
            .collect();
        build(n, &edges)
    }

    /// GPU level-0 local moving against CPU Coloured on a real graph, read
    /// from `BIXVERSE_LEIDEN_GRAPH` (skipped when unset). Modularity within 1%
    /// of the CPU's; prints the timings.
    #[test]
    // 829k-node harmony sNN, 9.5M edges: about a minute
    #[cfg(feature = "large-test")]
    fn test_leiden_gpu_large_graph() {
        let Some(client) = try_client() else { return };
        let Ok(path) = std::env::var("BIXVERSE_LEIDEN_GRAPH") else {
            return;
        };
        let g = read_edge_list(&path);
        println!("n {}", g.get_node_number());
        println!(
            "| algo | CPU Coloured s | GPU s | GPU upload s | GPU moving s | GPU rest s | Q CPU | Q GPU |"
        );
        for (name, refine, iters) in [("Louvain", false, 1), ("Leiden", true, 2)] {
            let p = LeidenParams::new(1.0, 0.01, iters, refine, 42);
            let cpu_p = LeidenParams {
                local_moving: LocalMoving::Coloured,
                ..p
            };
            let t = std::time::Instant::now();
            let cpu = community_detection(&g, &cpu_p, usize::MAX, true).unwrap();
            let t_cpu = t.elapsed().as_secs_f64();

            // the first run compiles the kernels
            let mut timed = TimedMover {
                inner: LeidenGpuMover::new(&client),
                prepare: 0.0,
                moving: 0.0,
            };
            community_detection_with(&g, &p, usize::MAX, true, &mut timed).unwrap();
            let mut timed = TimedMover {
                inner: LeidenGpuMover::new(&client),
                prepare: 0.0,
                moving: 0.0,
            };
            let t = std::time::Instant::now();
            let gpu = community_detection_with(&g, &p, usize::MAX, true, &mut timed).unwrap();
            let t_gpu = t.elapsed().as_secs_f64();

            let (q_cpu, q_gpu) = (modularity(&g, &cpu, 1.0), modularity(&g, &gpu, 1.0));
            println!(
                "| {name} | {t_cpu:.2} | {t_gpu:.2} | {:.2} | {:.2} | {:.2} | {q_cpu:.4} | {q_gpu:.4} |",
                timed.prepare,
                timed.moving,
                t_gpu - timed.prepare - timed.moving
            );
            assert!(
                q_gpu > q_cpu - 0.01 * q_cpu.abs(),
                "{name}: gpu {q_gpu} cpu {q_cpu}"
            );
        }
    }
}
