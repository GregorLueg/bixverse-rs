//! Graph community detection algorithms

use rand::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;
use std::time::Instant;

use crate::prelude::*;

/////////////
// Louvain //
/////////////

/// Louvain community detection
///
/// This version works on sparse graphs
///
/// ### Params
///
/// * `graph` - Undirected sparse graph
/// * `resolution` - Resolution parameter for the Louvain clustering
/// * `iter` - Numbers of iterations for the algorithm
/// * `seed` - Seed for reproducibility purposes
///
/// ### Returns
///
/// Vector of communitiies
pub fn louvain_sparse_graph<T>(
    graph: &SparseGraph<T>,
    resolution: T,
    max_iter: usize,
    seed: usize,
) -> Vec<usize>
where
    T: BixverseFloat + Clone + std::iter::Sum,
{
    assert!(
        !graph.is_directed(),
        "Louvain does not work for directed graphs!"
    );
    let n = graph.get_node_number();
    if n == 0 {
        return Vec::new();
    }
    let mut rng = StdRng::seed_from_u64(seed as u64);

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
    let mut node_order: Vec<u32> = (0..n as u32).collect();

    let epsilon = T::from_f64(1e-10).unwrap();

    for _ in 0..max_iter {
        let mut move_count = 0;
        node_order.shuffle(&mut rng);

        for &node in &node_order {
            let node_idx = node as usize;
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
                move_count += 1;
            }
        }

        if move_count == 0 {
            break;
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

    communities.iter().map(|&c| c as usize).collect()
}

//////////////
// Walktrap //
//////////////

/// Type of linkage to use for the WalkTrap community detection algorithm
#[derive(Clone, Copy)]
pub enum Linkage {
    /// Average Linkage
    Average,
    /// Complete linkage
    Complete,
}

/// Parse the linkage distance
///
/// ### Params
///
/// * `s` - The string to parse
///
/// ### Return
///
/// The Option of the `LinkageDist`
pub fn parse_linkage_distance(s: &str) -> Option<Linkage> {
    match s.to_lowercase().as_str() {
        "average" => Some(Linkage::Average),
        "complete" => Some(Linkage::Complete),
        _ => None,
    }
}

/////////////
// Helpers //
/////////////

/// Compute transition probability matrix
///
/// ### Params
///
/// * `graph` - SparseGraph structure
///
/// ### Returns
///
/// The transition probabilities from a given node to the others with their
/// weights
fn compute_transition_matrix<T>(graph: &SparseGraph<T>) -> Vec<Vec<(usize, T)>>
where
    T: BixverseFloat + std::iter::Sum,
{
    let n = graph.get_node_number();

    (0..n)
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
        .collect()
}

/// Compute distance based on random walk distributions
///
/// ### Params
///
/// * `all_probs` - Probability vectors for all nodes
/// * `i` - Index of node i
/// * `j` - Index of node j
/// * `degrees` - Precomputed degrees for all nodes
///
/// ### Returns
///
/// Distance
#[inline(always)]
fn walk_distance<T>(all_probs: &[Vec<T>], i: usize, j: usize, degrees: &[T]) -> T
where
    T: BixverseFloat,
{
    let n = all_probs[0].len();
    let mut dist_sq = T::zero();
    let epsilon = T::from_f64(1e-10).unwrap();

    for k in 0..n {
        let deg = degrees[k];
        if deg > epsilon {
            let diff = all_probs[i][k] - all_probs[j][k];
            dist_sq += (diff * diff) / deg;
        }
    }

    dist_sq.sqrt()
}

/// Compute random walk distances between all pairs within neighbourhood
///
/// ### Params
///
/// * `graph` - The SparseGraph
/// * `transition_probs` - Pre-calculated transition probabilities
/// * `walk_length` - Length of the walk
///
/// ### Returns
///
/// HashMap with the walk distances between the different points
fn compute_walk_distances<T>(
    graph: &SparseGraph<T>,
    transition_probs: &[Vec<(usize, T)>],
    walk_length: usize,
) -> FxHashMap<(usize, usize), T>
where
    T: BixverseFloat + std::iter::Sum,
{
    let n = graph.get_node_number();

    // Precompute degrees once
    let degrees: Vec<T> = (0..n)
        .map(|k| {
            let (_, weights) = graph.get_neighbours(k);
            weights.iter().copied().sum()
        })
        .collect();

    let epsilon = T::from_f64(1e-10).unwrap();
    let threshold = T::from_f64(1e6).unwrap();

    // Compute all probability distributions
    let all_probs: Vec<Vec<T>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut probs = vec![T::zero(); n];
            let mut new_probs = vec![T::zero(); n];
            probs[i] = T::one();

            for _ in 0..walk_length {
                for p in &mut new_probs {
                    *p = T::zero();
                }
                for node in 0..n {
                    let p = probs[node];
                    if p > epsilon {
                        for &(neighbour, trans_p) in &transition_probs[node] {
                            new_probs[neighbour] += p * trans_p;
                        }
                    }
                }
                std::mem::swap(&mut probs, &mut new_probs);
            }
            probs
        })
        .collect();

    // Compute pairwise distances (upper triangle only)
    let estimated_capacity = (n * (n - 1)) / 2;
    let all_distances: Vec<_> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            (i + 1..n)
                .filter_map(|j| {
                    let dist = walk_distance(&all_probs, i, j, &degrees);
                    if dist < threshold {
                        Some(((i, j), dist))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut result = FxHashMap::with_capacity_and_hasher(estimated_capacity, Default::default());
    result.extend(all_distances);
    result
}

/// Compute distance between two communities
///
/// ### Params
///
/// * `node_distances` - HashMap with all of the distances
/// * `comm_1` - Indices of members in community 1
/// * `comm_2` - Indices of members in community 2
/// * `linkage` - Linkage type
///
/// ### Returns
///
/// Distance between the members of the two communities
fn compute_community_distance<T>(
    node_distances: &FxHashMap<(usize, usize), T>,
    comm_1: &[usize],
    comm_2: &[usize],
    linkage: Linkage,
) -> T
where
    T: BixverseFloat,
{
    let fallback = T::from_f64(1e10).unwrap();

    match linkage {
        Linkage::Average => {
            let mut sum = T::zero();
            let mut count = 0;

            for &i in comm_1 {
                for &j in comm_2 {
                    let key = if i < j { (i, j) } else { (j, i) };
                    if let Some(&d) = node_distances.get(&key) {
                        sum += d;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                sum / T::from_usize(count).unwrap()
            } else {
                fallback
            }
        }
        Linkage::Complete => {
            let mut max_dist = T::zero();

            for &i in comm_1 {
                for &j in comm_2 {
                    let key = if i < j { (i, j) } else { (j, i) };
                    if let Some(&d) = node_distances.get(&key) {
                        max_dist = max_dist.max(d);
                    }
                }
            }

            if max_dist > T::zero() {
                max_dist
            } else {
                fallback
            }
        }
    }
}

/// WalkTrap community detection
///
/// ### Params
///
/// * `graph` - The SparseGraph.
/// * `walk_length` - The walk length for the random walkers.
/// * `num_clusters` - Number of communities to return.
/// * `linkage_dist` - The type of Linkage distance to use. One of `"average"`
///   or `"complete"`.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// The community memberships
pub fn walktrap_sparse_graph<T>(
    graph: &SparseGraph<T>,
    walk_length: usize,
    num_clusters: usize,
    linkage_dist: &str,
    verbose: bool,
) -> Vec<usize>
where
    T: BixverseFloat + std::iter::Sum,
{
    let n = graph.get_node_number();
    if n == 0 {
        return Vec::new();
    }
    if n <= num_clusters {
        return (0..n).collect();
    }

    let walktrap_start = Instant::now();

    // transition probabilities
    let start_transition_probs = Instant::now();

    let transition_probs = compute_transition_matrix(graph);

    let end_transition_probs = start_transition_probs.elapsed();

    if verbose {
        println!(
            "Calculated transition probabilities: {:.2?}",
            end_transition_probs
        );
    }

    // distance calculations
    let start_distance_calc = Instant::now();

    let rw_distances = compute_walk_distances(graph, &transition_probs, walk_length);

    let end_distance_calc = start_distance_calc.elapsed();

    if verbose {
        println!(
            "Calculated Random Walk distances: {:.2?}",
            end_distance_calc
        );
    }

    let linkage_dist = parse_linkage_distance(linkage_dist).unwrap_or(Linkage::Complete);

    let start_linkage = Instant::now();

    let mut community_map: Vec<usize> = (0..n).collect();
    let mut community_sizes: Vec<usize> = vec![1; n];
    let mut active_communities: Vec<bool> = vec![true; n];
    let mut community_generation: Vec<usize> = vec![0; n];
    let mut num_active = n;

    let mut distances: FxHashMap<(usize, usize), T> =
        FxHashMap::with_capacity_and_hasher(rw_distances.len(), Default::default());

    let mut merge_queue: BinaryHeap<(OrderedFloat<T>, usize, usize, usize, usize)> =
        BinaryHeap::with_capacity(rw_distances.len());

    for (&(i, j), &dist) in &rw_distances {
        if dist.is_finite() {
            distances.insert((i, j), dist);
            merge_queue.push((
                OrderedFloat(dist),
                i,
                j,
                community_generation[i],
                community_generation[j],
            ));
        }
    }

    let mut community_members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    while num_active > num_clusters {
        let (dist, mut c1, mut c2) = loop {
            if let Some((d, a, b, gen_a, gen_b)) = merge_queue.pop() {
                if active_communities[a]
                    && active_communities[b]
                    && community_generation[a] == gen_a
                    && community_generation[b] == gen_b
                {
                    break (d.0, a, b);
                }
            } else {
                break (T::infinity(), 0, 0);
            }
        };

        if !dist.is_finite() {
            break;
        }

        if c1 > c2 {
            std::mem::swap(&mut c1, &mut c2);
        }

        let new_size = community_sizes[c1] + community_sizes[c2];

        let mut new_members = std::mem::take(&mut community_members[c1]);
        new_members.extend_from_slice(&community_members[c2]);

        active_communities[c1] = false;
        active_communities[c2] = false;
        num_active -= 1;

        community_sizes.push(new_size);
        active_communities.push(true);
        community_members.push(new_members);
        community_generation.push(0);

        let new_comm = community_sizes.len() - 1;

        for &member in &community_members[new_comm] {
            community_map[member] = new_comm;
        }

        let active_comms: Vec<usize> = (0..active_communities.len())
            .filter(|&i| active_communities[i] && i != new_comm)
            .collect();

        let new_distances: Vec<_> = active_comms
            .par_iter()
            .filter_map(|&other| {
                let dist_new = compute_community_distance(
                    &rw_distances,
                    &community_members[new_comm],
                    &community_members[other],
                    linkage_dist,
                );

                if dist_new.is_finite() {
                    let key = if new_comm < other {
                        (new_comm, other)
                    } else {
                        (other, new_comm)
                    };
                    Some((key, dist_new, new_comm, other))
                } else {
                    None
                }
            })
            .collect();

        for (key, dist_new, comm1, comm2) in new_distances {
            distances.insert(key, dist_new);
            merge_queue.push((
                OrderedFloat(dist_new),
                comm1,
                comm2,
                community_generation[comm1],
                community_generation[comm2],
            ));
        }
    }

    let end_linkage = start_linkage.elapsed();

    if verbose {
        println!("Finished merging the communities: {:.2?}", end_linkage);
    }

    let mut final_labels = vec![0; n];
    let mut label_map: FxHashMap<usize, usize> =
        FxHashMap::with_capacity_and_hasher(num_clusters, Default::default());
    let mut next_label = 0;

    for node in 0..n {
        let comm = community_map[node];
        let label = *label_map.entry(comm).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        final_labels[node] = label;
    }

    let walktrap_end = walktrap_start.elapsed();

    if verbose {
        println!(
            "Finished WalkTrap communitie detection: {:.2?}",
            walktrap_end
        );
    }

    final_labels
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::sparse::CompressedSparseData2;

    fn build_barbell_graph() -> SparseGraph<f64> {
        // Two triangles (0,1,2) and (3,4,5) connected by edge (2,3)
        // Edges: 0-1, 1-2, 2-0, 3-4, 4-5, 5-3, 2-3
        // Symmetric CSR representation:
        let indptr = vec![0, 2, 4, 7, 10, 12, 14];
        let indices = vec![
            1, 2, // 0's neighbors
            0, 2, // 1's neighbors
            0, 1, 3, // 2's neighbors
            2, 4, 5, // 3's neighbors
            3, 5, // 4's neighbors
            3, 4, // 5's neighbors
        ];
        let data = vec![1.0; 14];

        let csr =
            CompressedSparseData2::<f64, f64>::new_csr(&data, &indices, &indptr, None, (6, 6));
        SparseGraph::new(6, csr, false)
    }

    #[test]
    fn test_louvain_barbell() {
        let graph = build_barbell_graph();
        let comms = louvain_sparse_graph(&graph, 1.0, 10, 42);

        // 0, 1, 2 should share a community
        assert_eq!(comms[0], comms[1]);
        assert_eq!(comms[1], comms[2]);
        // 3, 4, 5 should share a different community
        assert_eq!(comms[3], comms[4]);
        assert_eq!(comms[4], comms[5]);
        assert_ne!(comms[2], comms[3]);
    }

    #[test]
    fn test_walktrap_barbell() {
        let graph = build_barbell_graph();
        // Ask for exactly 2 clusters
        let comms = walktrap_sparse_graph(&graph, 3, 2, "complete", false);

        assert_eq!(comms[0], comms[1]);
        assert_eq!(comms[1], comms[2]);
        assert_eq!(comms[3], comms[4]);
        assert_eq!(comms[4], comms[5]);
        assert_ne!(comms[2], comms[3]);
    }
}
