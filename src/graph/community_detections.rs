//! Graph community detection algorithms. Includes Louvain and WalkTrap
//! implementations.

use ann_search_rs::prelude::SimdDistance;
use ann_search_rs::utils::dist::euclidean_distance_static;
use rand::prelude::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BinaryHeap;
use std::collections::VecDeque;
use std::time::Instant;

use crate::core::math::sparse::coo_to_csr;
use crate::prelude::*;

/////////////////////
// General helpers //
/////////////////////

/// Helper function to remap communities by size
///
/// The non-determinism due to threading etc. can be a huge annoyance in
/// single cell. This function remaps the clusters by size, ensuring more (not
/// complete) reproducibility.
///
/// ### Params
///
/// * `labels` - The membership labels
///
/// ### Returns
///
/// Remapped labels with `0` -> biggest community by membership, `1` -> second
/// biggest, etc.
fn remap_communities_by_size(labels: &[usize]) -> Vec<usize> {
    let n = labels.len();
    let mut counts: FxHashMap<usize, usize> = FxHashMap::default();
    for &l in labels {
        *counts.entry(l).or_insert(0) += 1;
    }

    let mut sorted: Vec<(usize, usize)> = counts.into_iter().collect();
    sorted.sort_unstable_by_key(|a| std::cmp::Reverse(a.1));

    let mut remap: FxHashMap<usize, usize> = FxHashMap::default();
    for (new_label, (old_label, _)) in sorted.into_iter().enumerate() {
        remap.insert(old_label, new_label);
    }

    (0..n).map(|i| remap[&labels[i]]).collect()
}

/////////////
// Louvain //
/////////////

/////////////
// Helpers //
/////////////

/// One Phase 1 pass with pruning.
///
/// Maintains a queue of dirty nodes. A node is dirty when first added or when
/// one of its neighbours has just moved community. Drains until empty.
///
/// ### Params
///
/// * `graph` - The [`SparseGraph`] - must be undirected!
/// * `resolution` - The resolution parameter to use
/// * `max_iter` - The maximum iterations to run the algorithm for. In this
///   case, it safeguards against degenerated graphs and should not trigger.
/// * `rng` - The random number generator
///
/// ### Returns
///
/// Returns `(contiguous labels in [0, n_distinct), count)`.
fn louvain_one_level<T>(
    graph: &SparseGraph<T>,
    resolution: T,
    max_iter: usize,
    rng: &mut StdRng,
) -> (Vec<usize>, usize)
where
    T: BixverseFloat + BixverseNumeric + Clone + std::iter::Sum,
{
    let n = graph.get_node_number();

    let m: T = (0..n)
        .map(|i| graph.get_neighbours(i).1.iter().copied().sum::<T>())
        .sum::<T>()
        / T::from_f64(2.0).unwrap();

    let two_m = T::from_f64(2.0).unwrap() * m;
    let res_over_two_m = resolution / two_m;

    let mut degrees = vec![T::zero(); n];
    for i in 0..n {
        degrees[i] = graph.get_neighbours(i).1.iter().copied().sum();
    }

    let mut communities: Vec<u32> = (0..n as u32).collect();
    let mut comm_degree_sums = degrees.clone();
    let mut neighbour_weights = vec![T::zero(); n];
    let mut comm_active = vec![false; n];
    let mut active_comms = Vec::with_capacity(256);

    let epsilon = T::from_f64(1e-10).unwrap();

    let mut initial_order: Vec<u32> = (0..n as u32).collect();
    initial_order.shuffle(rng);
    let mut queue: VecDeque<u32> = initial_order.into_iter().collect();
    let mut in_queue = vec![true; n];

    let max_evals = max_iter.saturating_mul(n);
    let mut evals = 0usize;

    while let Some(node) = queue.pop_front() {
        if evals >= max_evals {
            break;
        }
        evals += 1;

        let node_idx = node as usize;
        in_queue[node_idx] = false;

        let current_comm = communities[node_idx] as usize;
        let k_i = degrees[node_idx];
        let k_i_scaled = k_i * res_over_two_m;

        let (neighbours, weights) = graph.get_neighbours(node_idx);

        for (&neighbour, &weight) in neighbours.iter().zip(weights.iter()) {
            let comm = communities[neighbour] as usize;
            if !comm_active[comm] {
                comm_active[comm] = true;
                active_comms.push(comm);
            }
            neighbour_weights[comm] += weight;
        }

        let mut best_comm = current_comm;
        let mut best_delta = T::zero();

        for &comm in &active_comms {
            if comm != current_comm {
                let delta = neighbour_weights[comm] - k_i_scaled * comm_degree_sums[comm];
                if delta > best_delta {
                    best_delta = delta;
                    best_comm = comm;
                }
            }
        }

        for &comm in &active_comms {
            neighbour_weights[comm] = T::zero();
            comm_active[comm] = false;
        }
        active_comms.clear();

        if best_comm != current_comm && best_delta > epsilon {
            communities[node_idx] = best_comm as u32;
            comm_degree_sums[current_comm] -= k_i;
            comm_degree_sums[best_comm] += k_i;

            for &nb in neighbours {
                if nb != node_idx && !in_queue[nb] {
                    queue.push_back(nb as u32);
                    in_queue[nb] = true;
                }
            }
        }
    }

    let mut comm_map = vec![u32::MAX; n];
    let mut label = 0u32;
    for c in &mut communities {
        let idx = *c as usize;
        if comm_map[idx] == u32::MAX {
            comm_map[idx] = label;
            label += 1;
        }
        *c = comm_map[idx];
    }

    let n_distinct = label as usize;
    (
        communities.iter().map(|&c| c as usize).collect(),
        n_distinct,
    )
}

/// Aggregate fine nodes by community into a coarser graph.
///
/// Internal community edges become self-loops on the super-node (with weight
/// 2x the internal edge weight, as both CSR directions accumulate). External
/// edges sum across all member-to-member connections. Total weight m is
/// preserved across aggregation.
///
/// ### Params
///
/// * `graph` - The [`SparseGraph`] - must be undirected!
/// * `communities` - The community membership from the finer level.
/// * `n_comms` - Number of communities
///
/// ### Returns
///
/// Coarsed sparse graph
fn aggregate_graph<T>(
    graph: &SparseGraph<T>,
    communities: &[usize],
    n_comms: usize,
) -> SparseGraph<T>
where
    T: BixverseFloat + BixverseNumeric + Clone + std::iter::Sum,
{
    let mut nodes_per_comm: Vec<Vec<usize>> = vec![Vec::new(); n_comms];
    for (i, &c) in communities.iter().enumerate() {
        nodes_per_comm[c].push(i);
    }

    let mut acc = vec![T::zero(); n_comms];
    let mut touched = vec![false; n_comms];
    let mut active: Vec<usize> = Vec::with_capacity(64);

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for ci in 0..n_comms {
        for &fine_i in &nodes_per_comm[ci] {
            let (neighbours, weights) = graph.get_neighbours(fine_i);
            for (&j, &w) in neighbours.iter().zip(weights.iter()) {
                let cj = communities[j];
                if !touched[cj] {
                    touched[cj] = true;
                    active.push(cj);
                }
                acc[cj] += w;
            }
        }

        for &cj in &active {
            rows.push(ci);
            cols.push(cj);
            vals.push(acc[cj]);
            acc[cj] = T::zero();
            touched[cj] = false;
        }
        active.clear();
    }

    let csr = coo_to_csr(&rows, &cols, &vals, (n_comms, n_comms));
    SparseGraph::new(n_comms, csr, false)
}

//////////
// Main //
//////////

/// Louvain community detection on a [`SparseGraph`].
///
/// ### Params
///
/// * `graph` - The [`SparseGraph`]
/// * `resolution` - Resolution parameter for the Louvain clustering
/// * `max_iter` - The maximum iterations to run the algorithm for. In this
///   case, it safeguards against degenerated graphs and should not trigger.
/// * `multi_level` - If `true`, run the full multi-level Louvain (Phase 1 +
///   Phase 2 aggregation, repeated). If `false`, run only one Phase 1 pass
///   (matches Phenograph / the original doubletdetection behaviour).
/// * `seed` - Seed for reproducibility purposes
///
/// ### Returns
///
/// Vector of communities
pub fn louvain_sparse_graph<T>(
    graph: &SparseGraph<T>,
    resolution: T,
    max_iter: usize,
    multi_level: bool,
    seed: usize,
) -> Result<Vec<usize>, BixverseErrors>
where
    T: BixverseFloat + BixverseNumeric + Clone + std::iter::Sum,
{
    if graph.is_directed() {
        return Err(BixverseErrors::GraphDirectedError);
    }

    let n_orig = graph.get_node_number();
    if n_orig == 0 {
        return Ok(Vec::new());
    }

    let mut rng = StdRng::seed_from_u64(seed as u64);
    let mut node_to_super: Vec<usize> = (0..n_orig).collect();

    let (comms, mut n_super) = louvain_one_level(graph, resolution, max_iter, &mut rng);
    for s in node_to_super.iter_mut() {
        *s = comms[*s];
    }

    if !multi_level || n_super == n_orig {
        return Ok(remap_communities_by_size(&node_to_super));
    }

    let mut current_graph = aggregate_graph(graph, &comms, n_super);

    loop {
        let (comms, new_n_super) =
            louvain_one_level(&current_graph, resolution, max_iter, &mut rng);
        for s in node_to_super.iter_mut() {
            *s = comms[*s];
        }
        if new_n_super == n_super {
            break;
        }
        n_super = new_n_super;
        current_graph = aggregate_graph(&current_graph, &comms, n_super);
    }

    Ok(remap_communities_by_size(&node_to_super))
}

//////////////
// Walktrap //
//////////////

/// Compute random-walk probability vectors for all nodes
///
/// For each node i, simulates a `walk_length`-step random walk starting at i
/// and records the landing probabilities. The resulting vector is then scaled
/// by the inverse square root of each node's degree, matching the Walktrap
/// normalisation from Pons & Latapy 2005.
///
/// Transition probabilities are weight-proportional: `p(i→j) = w(i,j) / deg(i)`.
/// Isolated nodes (zero degree) yield an all-zero probability vector.
///
/// ### Params
///
/// * `graph` - Sparse weighted graph
/// * `walk_length` - Number of steps per random walk
///
/// ### Returns
///
/// Flat row-major buffer of shape `(n × n)` where row i holds the scaled
/// landing probabilities `q[i, j] = P(walk from i lands at j) / sqrt(deg(j))`
fn compute_walk_probabilities<T>(graph: &SparseGraph<T>, walk_length: usize) -> Vec<T>
where
    T: BixverseFloat + std::iter::Sum + Send + Sync,
{
    let n = graph.get_node_number();
    let epsilon = T::from_f64(1e-10).unwrap();

    let inv_sqrt_deg: Vec<T> = (0..n)
        .map(|k| {
            let (_, w) = graph.get_neighbours(k);
            let d: T = w.iter().copied().sum();
            if d > T::zero() {
                T::one() / d.sqrt()
            } else {
                T::zero()
            }
        })
        .collect();

    let transition_probs: Vec<Vec<(usize, T)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (neighbours, weights) = graph.get_neighbours(i);
            let degree: T = weights.iter().copied().sum();
            if degree > T::zero() {
                neighbours
                    .iter()
                    .zip(weights.iter())
                    .map(|(&j, &w)| (j, w / degree))
                    .collect()
            } else {
                Vec::new()
            }
        })
        .collect();

    let mut q_flat: Vec<T> = vec![T::zero(); n * n];
    q_flat.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let mut probs = vec![T::zero(); n];
        let mut new_probs = vec![T::zero(); n];
        probs[i] = T::one();
        for _ in 0..walk_length {
            for p in new_probs.iter_mut() {
                *p = T::zero();
            }
            for node in 0..n {
                let p = probs[node];
                if p > epsilon {
                    for &(nb, tp) in &transition_probs[node] {
                        new_probs[nb] += p * tp;
                    }
                }
            }
            std::mem::swap(&mut probs, &mut new_probs);
        }
        for k in 0..n {
            row[k] = probs[k] * inv_sqrt_deg[k];
        }
    });

    q_flat
}

/// Compute Ward's linkage criterion between two communities
///
/// Returns the increase in total within-cluster variance that would result from
/// merging communities A and B, given their squared centroid distance:
///
/// ```Ward(A, B) = (n_a · n_b) / (n_a + n_b) · d²(A, B)```
///
/// ### Params
///
/// * `d_sq` - Squared distance between the community centroids
/// * `n_a` - Size of community A
/// * `n_b` - Size of community B
///
/// ### Returns
///
/// Ward criterion value (lower means a cheaper merge)
#[inline(always)]
fn ward_criterion<T>(d_sq: T, n_a: usize, n_b: usize) -> T
where
    T: BixverseFloat,
{
    let na = T::from_usize(n_a).unwrap();
    let nb = T::from_usize(n_b).unwrap();
    (na * nb) / (na + nb) * d_sq
}

/// Resolve final community labels from a union-find parent array
///
/// Follows each node's parent chain to its root and assigns a contiguous
/// integer label `0..num_communities` in traversal order. The result is a flat
/// label per node with no gaps.
///
/// ### Params
///
/// * `n` - Number of original nodes
/// * `parent` - Union-find parent array; `parent[x] == x` denotes a root
///
/// ### Returns
///
/// Vector of community labels, one per node
fn finalise_labels(n: usize, parent: &[usize]) -> Vec<usize> {
    let mut labels = vec![0usize; n];
    let mut label_map: FxHashMap<usize, usize> = FxHashMap::default();
    let mut next = 0usize;
    for node in 0..n {
        let mut x = node;
        while parent[x] != x {
            x = parent[x];
        }
        let label = *label_map.entry(x).or_insert_with(|| {
            let l = next;
            next += 1;
            l
        });
        labels[node] = label;
    }
    labels
}

/// Walktrap community detection (Pons & Latapy 2005)
///
/// Identifies communities by agglomerative clustering of random-walk probs
/// vectors. Nodes that tend to co-occur on short random walks are grouped
/// together.
///
/// Algorithm outline:
///
/// 1. Compute scaled walk-probability vectors for all nodes
/// 2. Initialise each node as its own community
/// 3. Greedily merge the adjacent pair with the smallest Ward criterion
///    until `num_clusters` communities remain
/// 4. Resolve labels via union-find
///
/// Merged community vectors are updated as the weighted average:
/// `q[A∪B] = (n_a · q[A] + n_b · q[B]) / (n_a + n_b)`
///
/// ### Params
///
/// * `graph` - Sparse weighted graph to cluster
/// * `walk_length` - Number of steps for the random walks
/// * `num_clusters` - Target number of communities to return
/// * `verbose` - Print timing information for each stage
///
/// ### Returns
///
/// Vector of community labels (0-indexed, contiguous), one per node
pub fn walktrap_sparse_graph<T>(
    graph: &SparseGraph<T>,
    walk_length: usize,
    num_clusters: usize,
    verbose: bool,
) -> Vec<usize>
where
    T: BixverseFloat + std::iter::Sum + Send + Sync + SimdDistance,
{
    let n = graph.get_node_number();
    if n == 0 {
        return Vec::new();
    }
    if n <= num_clusters {
        return (0..n).collect();
    }

    let walktrap_start = Instant::now();

    let t0 = Instant::now();
    let q_flat = compute_walk_probabilities(graph, walk_length);
    if verbose {
        println!("Calculated walk probabilities: {:.2?}", t0.elapsed());
    }

    let mut comm_q: Vec<Option<Vec<T>>> = (0..n)
        .map(|i| Some(q_flat[i * n..(i + 1) * n].to_vec()))
        .collect();
    drop(q_flat);

    let mut comm_size: Vec<usize> = vec![1; n];
    let mut active: Vec<bool> = vec![true; n];
    let mut comm_parent: Vec<usize> = (0..n).collect();
    let mut adj: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];

    let mut edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        let (neighbours, _) = graph.get_neighbours(i);
        for &j in neighbours {
            if i != j {
                adj[i].insert(j);
                if i < j {
                    edges.push((i, j));
                }
            }
        }
    }

    let t1 = Instant::now();
    let initial: Vec<(RevOrderedFloat<T>, usize, usize)> = edges
        .par_iter()
        .map(|&(i, j)| {
            let qi = comm_q[i].as_ref().unwrap();
            let qj = comm_q[j].as_ref().unwrap();
            let d_sq = euclidean_distance_static(qi, qj);
            let crit = ward_criterion(d_sq, 1, 1);
            (RevOrderedFloat(crit), i, j)
        })
        .collect();

    let mut heap: BinaryHeap<(RevOrderedFloat<T>, usize, usize)> =
        BinaryHeap::with_capacity(initial.len() * 2);
    for entry in initial {
        heap.push(entry);
    }
    if verbose {
        println!("Computed initial distances: {:.2?}", t1.elapsed());
    }

    let t2 = Instant::now();
    let mut num_active = n;

    while num_active > num_clusters {
        let (a, b) = loop {
            match heap.pop() {
                Some((_, i, j)) => {
                    if active[i] && active[j] {
                        break (i, j);
                    }
                }
                None => {
                    if verbose {
                        println!(
                            "Heap exhausted with {} communities active (graph likely disconnected)",
                            num_active
                        );
                        println!("Finished merging communities: {:.2?}", t2.elapsed());
                        println!(
                            "Finished WalkTrap community detection: {:.2?}",
                            walktrap_start.elapsed()
                        );
                    }
                    return finalise_labels(n, &comm_parent);
                }
            }
        };

        let n_a = comm_size[a];
        let n_b = comm_size[b];
        let n_c = n_a + n_b;
        let na_t = T::from_usize(n_a).unwrap();
        let nb_t = T::from_usize(n_b).unwrap();
        let nc_t = T::from_usize(n_c).unwrap();

        let mut new_q = vec![T::zero(); n];
        {
            let qa = comm_q[a].as_ref().unwrap();
            let qb = comm_q[b].as_ref().unwrap();
            for k in 0..n {
                new_q[k] = (na_t * qa[k] + nb_t * qb[k]) / nc_t;
            }
        }

        let mut new_adj: FxHashSet<usize> = FxHashSet::default();
        for &k in &adj[a] {
            if k != b {
                new_adj.insert(k);
            }
        }
        for &k in &adj[b] {
            if k != a {
                new_adj.insert(k);
            }
        }

        active[a] = false;
        active[b] = false;
        comm_q[a] = None;
        comm_q[b] = None;
        adj[a] = FxHashSet::default();
        adj[b] = FxHashSet::default();

        let c = comm_q.len();
        comm_q.push(Some(new_q));
        comm_size.push(n_c);
        active.push(true);
        comm_parent.push(c);
        comm_parent[a] = c;
        comm_parent[b] = c;
        adj.push(new_adj.clone());

        for &k in &new_adj {
            adj[k].remove(&a);
            adj[k].remove(&b);
            adj[k].insert(c);
        }

        let new_adj_vec: Vec<usize> = new_adj.into_iter().collect();
        let new_criteria: Vec<(RevOrderedFloat<T>, usize, usize)> = new_adj_vec
            .par_iter()
            .map(|&k| {
                let qc = comm_q[c].as_ref().unwrap();
                let qk = comm_q[k].as_ref().unwrap();
                let d_sq = euclidean_distance_static(qc, qk);
                let crit = ward_criterion(d_sq, n_c, comm_size[k]);
                (RevOrderedFloat(crit), c, k)
            })
            .collect();

        for entry in new_criteria {
            heap.push(entry);
        }

        num_active -= 1;
    }

    if verbose {
        println!("Finished merging communities: {:.2?}", t2.elapsed());
        println!(
            "Finished WalkTrap community detection: {:.2?}",
            walktrap_start.elapsed()
        );
    }

    let labels = finalise_labels(n, &comm_parent);

    remap_communities_by_size(&labels)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::sparse::CompressedSparseData2;

    fn build_barbell_graph() -> SparseGraph<f64> {
        let indptr = vec![0, 2, 4, 7, 10, 12, 14];
        let indices = vec![1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4];
        let data = vec![1.0; 14];
        let csr =
            CompressedSparseData2::<f64, f64>::new_csr(&data, &indices, &indptr, None, (6, 6));
        SparseGraph::new(6, csr, false)
    }

    fn build_two_cliques_graph() -> SparseGraph<f64> {
        // K4 {0..3} + K4 {4..7} bridged by edge 3-4
        let mut nbrs: Vec<Vec<usize>> = vec![Vec::new(); 8];
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    nbrs[i].push(j);
                }
            }
        }
        for i in 4..8 {
            for j in 4..8 {
                if i != j {
                    nbrs[i].push(j);
                }
            }
        }
        nbrs[3].push(4);
        nbrs[4].push(3);

        let mut indptr = vec![0usize];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for list in &mut nbrs {
            list.sort();
            for &j in list.iter() {
                indices.push(j);
                data.push(1.0);
            }
            indptr.push(indices.len());
        }
        let csr =
            CompressedSparseData2::<f64, f64>::new_csr(&data, &indices, &indptr, None, (8, 8));
        SparseGraph::new(8, csr, false)
    }

    #[test]
    fn test_louvain_barbell() {
        let graph = build_barbell_graph();
        let comms = louvain_sparse_graph(&graph, 1.0, 10, true, 42).unwrap();
        assert_eq!(comms[0], comms[1]);
        assert_eq!(comms[1], comms[2]);
        assert_eq!(comms[3], comms[4]);
        assert_eq!(comms[4], comms[5]);
        assert_ne!(comms[2], comms[3]);
    }

    #[test]
    fn test_walktrap_barbell() {
        let graph = build_barbell_graph();
        let comms = walktrap_sparse_graph(&graph, 3, 2, false);
        assert_eq!(comms[0], comms[1]);
        assert_eq!(comms[1], comms[2]);
        assert_eq!(comms[3], comms[4]);
        assert_eq!(comms[4], comms[5]);
        assert_ne!(comms[2], comms[3]);
    }

    #[test]
    fn test_walktrap_two_cliques() {
        let graph = build_two_cliques_graph();
        let comms = walktrap_sparse_graph(&graph, 4, 2, false);
        for i in 1..4 {
            assert_eq!(comms[i], comms[0]);
        }
        for i in 5..8 {
            assert_eq!(comms[i], comms[4]);
        }
        assert_ne!(comms[0], comms[4]);
    }
}
