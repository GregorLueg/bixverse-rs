//! Partition-based graph abstraction, the connectivity statistic behind PAGA.
//!
//! Coarse-grains a graph over a partition of its nodes and scores each pair of
//! partitions by how many edges actually run between them versus how many a
//! random null model predicts. The result is a small symmetric graph whose edge
//! weights read as confidence in a connection, capped at one.
//!
//! The input adjacency is read as **directed and binarised**: only the sparsity
//! pattern matters, stored values are ignored entirely. That is what the
//! reference does (it sets the kNN distance matrix's data to ones and builds a
//! directed igraph from it), and it lets callers hand in a
//! `CompressedSparseData2<u8>` and pay a byte per edge.
//!
//! Counting is `O(nnz)` into a dense `n_partitions²` matrix, so the whole thing
//! is one pass over the graph plus arithmetic on something the size of a
//! clustering.
//!
//! ### References
//!
//! Wolf, et al., Genome Biol., 2019. "PAGA: graph abstraction reconciles
//! clustering with trajectory inference through a topology preserving map of
//! single cells."

use rayon::prelude::*;

use crate::core::math::sparse::undirected_edges_to_csr;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Partition count above which edge counting drops to a single accumulator.
///
/// The parallel path keeps one dense `n_partitions²` `u64` accumulator per
/// rayon chunk. At 512 partitions that is 2 MB each, which is the most worth
/// paying for a pass that is only `O(nnz)` to begin with. Real clusterings sit
/// two orders of magnitude below this, so the sequential arm is a guard rather
/// than a path anyone takes.
const PARALLEL_PARTITION_LIMIT: usize = 512;

/// Largest partition count the dense coarsening accepts.
///
/// The counting matrix is `n_partitions²` `u64`, so 4096 partitions is already
/// 128 MB and the pair scan behind it is 16M cells. Graph abstraction is a
/// coarse-graining method; a partition vector this fine is a mis-specified
/// input rather than a workload worth a sparse rewrite, so it errors instead of
/// quietly allocating.
const MAX_PARTITIONS: usize = 4096;

/////////////
// Structs //
/////////////

/// The abstracted graph produced by [partition_connectivities].
pub struct PartitionConnectivity<T>
where
    T: Clone,
{
    /// Connectivities between partitions, `n_partitions` square, symmetric CSR
    /// with a zero diagonal. Stored values lie in `(0, 1]`.
    pub connectivities: CompressedSparseData2<T>,
    /// Nodes per partition, indexed by partition id. Partitions declared
    /// through `n_partitions` but never used keep a size of zero.
    pub sizes: Vec<usize>,
}

//////////
// Main //
//////////

/// Connectivity of a graph's partitions under a random-connection null model.
///
/// For partitions `a` and `b`, with `e[a]` the number of edges leaving `a`
/// (within-partition edges included), `s[a]` its node count and `n` the total
/// node count:
///
/// ```text
/// observed[a][b] = edges(a -> b) + edges(b -> a)
/// expected[a][b] = (e[a] * s[b] + e[b] * s[a]) / (n - 1)
/// conn[a][b]     = min(observed / expected, 1)
/// ```
///
/// The cap matters: the null over-estimates what "connected" means in real
/// data, so ratios above one are common and are not more informative than one.
/// See the reference's discussion.
///
/// Counting runs into a dense `n_partitions²` matrix whose **diagonal holds the
/// within-partition edge count**, which makes `e[a]` a plain row sum. The
/// diagonal is then dropped, mirroring the self-loop removal the reference gets
/// from igraph's `simplify`.
///
/// Every accumulation and the ratio itself run in `f64` before the cast to `T`.
/// Edge counts reach `10^8` on a million-node graph and the expected value is a
/// sum of two large products, so `f32` would lose the low bits of both.
///
/// ### Params
///
/// * `graph` - Square CSR adjacency, read as directed. **Stored values are
///   ignored**; only the sparsity pattern counts. Self loops are counted onto
///   the diagonal and then discarded, so they influence nothing but `e[a]`.
/// * `partitions` - Partition label per node, one per row of `graph`.
/// * `n_partitions` - Declared partition count. `None` derives it as
///   `max(label) + 1`, which drops trailing empty partitions; pass it
///   explicitly to keep unused levels of a factor.
///
/// ### Returns
///
/// The abstracted graph and the partition sizes, or an error when `graph` is
/// not a square CSR, the label count does not match the node count, a label
/// sits outside `n_partitions`, or `n_partitions` exceeds [MAX_PARTITIONS].
///
/// ### References
///
/// Wolf, et al., Genome Biol., 2019.
pub fn partition_connectivities<T, U>(
    graph: &CompressedSparseData2<U>,
    partitions: &[usize],
    n_partitions: Option<usize>,
) -> Result<PartitionConnectivity<T>, BixverseErrors>
where
    T: BixverseFloat + BixverseNumeric,
    U: Clone + Sync,
{
    let n = validate_square_csr(graph)?;
    if partitions.len() != n {
        return Err(BixverseErrors::CommunityAssignmentMismatch);
    }

    let k = match n_partitions {
        Some(k) => k,
        None => partitions.iter().copied().max().map_or(0, |m| m + 1),
    };
    if k > MAX_PARTITIONS {
        return Err(BixverseErrors::TooManyPartitions {
            n_partitions: k,
            max: MAX_PARTITIONS,
        });
    }
    if let Some(&label) = partitions.iter().find(|&&p| p >= k) {
        return Err(BixverseErrors::PartitionLabelOutOfRange {
            label,
            n_partitions: k,
        });
    }

    let mut sizes = vec![0usize; k];
    for &p in partitions {
        sizes[p] += 1;
    }

    if k == 0 {
        return Ok(PartitionConnectivity {
            connectivities: undirected_edges_to_csr::<T>(0, &[]),
            sizes,
        });
    }

    let counts = count_partition_edges(graph, partitions, k);

    // Row sums include the diagonal, which is exactly the reference's
    // `es_inner_cluster + inter_es.sum(axis=1)`.
    let out_edges: Vec<f64> = (0..k)
        .map(|a| counts[a * k..(a + 1) * k].iter().sum::<u64>() as f64)
        .collect();

    let edges = connectivity_edges::<T>(&counts, &out_edges, &sizes, n, k);

    Ok(PartitionConnectivity {
        connectivities: undirected_edges_to_csr(k, &edges),
        sizes,
    })
}

/////////////
// Helpers //
/////////////

/// Count directed edges between every ordered pair of partitions.
///
/// The accumulator is dense `k * k` in row-major order, with
/// `counts[a * k + b]` holding the number of stored edges from a node in `a` to
/// a node in `b`. The diagonal therefore holds the within-partition count.
///
/// Parallel below [PARALLEL_PARTITION_LIMIT] partitions: rows are split into
/// one chunk per thread and each chunk folds into its own accumulator, merged
/// in the reduce. Above the limit a per-thread copy costs more than the scan
/// saves, so a single accumulator runs sequentially.
///
/// ### Params
///
/// * `graph` - Square CSR adjacency, values ignored.
/// * `partitions` - Partition label per node.
/// * `k` - Partition count, at least one.
///
/// ### Returns
///
/// The dense `k * k` count matrix, row-major.
fn count_partition_edges<U>(
    graph: &CompressedSparseData2<U>,
    partitions: &[usize],
    k: usize,
) -> Vec<u64>
where
    U: Clone + Sync,
{
    let n = partitions.len();

    if k > PARALLEL_PARTITION_LIMIT {
        let mut counts = vec![0u64; k * k];
        for i in 0..n {
            let row = partitions[i] * k;
            for idx in graph.indptr[i] as usize..graph.indptr[i + 1] as usize {
                counts[row + partitions[graph.indices[idx] as usize]] += 1;
            }
        }
        return counts;
    }

    let chunk_len = n.div_ceil(rayon::current_num_threads().max(1)).max(1);

    (0..n)
        .into_par_iter()
        .with_min_len(chunk_len)
        .fold(
            || vec![0u64; k * k],
            |mut counts, i| {
                let row = partitions[i] * k;
                for idx in graph.indptr[i] as usize..graph.indptr[i + 1] as usize {
                    counts[row + partitions[graph.indices[idx] as usize]] += 1;
                }
                counts
            },
        )
        .reduce(
            || vec![0u64; k * k],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        )
}

/// Turn the edge counts into the upper-triangular connectivity edge list.
///
/// The diagonal is skipped rather than zeroed: only pairs with `a < b` are
/// visited, so the within-partition counts never reach the output. Pairs with
/// no edges between them are left out of the sparsity pattern entirely.
///
/// ### Params
///
/// * `counts` - Dense `k * k` directed edge counts, row-major.
/// * `out_edges` - Edges leaving each partition, the row sums of `counts`.
/// * `sizes` - Nodes per partition.
/// * `n` - Total node count.
/// * `k` - Partition count.
///
/// ### Returns
///
/// `(a, b, connectivity)` triples with `a < b`, ascending, ready for
/// [undirected_edges_to_csr].
fn connectivity_edges<T>(
    counts: &[u64],
    out_edges: &[f64],
    sizes: &[usize],
    n: usize,
    k: usize,
) -> Vec<(u32, u32, T)>
where
    T: BixverseFloat,
{
    // The null divides by `n - 1`. Below two nodes there are no cross-partition
    // edges to score anyway, so bail before the division rather than guarding
    // inside the loop.
    if n < 2 {
        return Vec::new();
    }

    let denominator = (n - 1) as f64;
    let mut edges = Vec::new();

    for a in 0..k {
        for b in (a + 1)..k {
            let observed = counts[a * k + b] + counts[b * k + a];
            if observed == 0 {
                continue;
            }

            let expected =
                (out_edges[a] * sizes[b] as f64 + out_edges[b] * sizes[a] as f64) / denominator;
            // An observed edge with a zero expectation cannot happen from a
            // consistent count matrix, but the reference scores it as full
            // confidence rather than dividing by zero.
            let connectivity = if expected != 0.0 {
                (observed as f64 / expected).min(1.0)
            } else {
                1.0
            };

            edges.push((a as u32, b as u32, T::from_f64(connectivity).unwrap()));
        }
    }

    edges
}

/// Check that a graph is a square CSR matrix and return its node count.
///
/// ### Params
///
/// * `graph` - The adjacency to validate.
///
/// ### Returns
///
/// The node count, or an error when the matrix is not CSR or not square.
fn validate_square_csr<U>(graph: &CompressedSparseData2<U>) -> Result<usize, BixverseErrors>
where
    U: Clone,
{
    if !graph.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }
    let (rows, cols) = graph.shape;
    if rows != cols {
        return Err(BixverseErrors::ShapeMismatch {
            expected: (rows, rows),
            got: (rows, cols),
        });
    }
    Ok(rows)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Build a directed binary CSR from an edge list.
    ///
    /// ### Params
    ///
    /// * `n` - Node count.
    /// * `edges` - Directed edges as `(from, to)`.
    ///
    /// ### Returns
    ///
    /// The CSR adjacency storing exactly the given directions.
    fn directed_csr(n: usize, edges: &[(usize, usize)]) -> CompressedSparseData2<u8> {
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &(a, b) in edges {
            rows[a].push(b as u32);
        }

        let mut indptr: Vec<u32> = vec![0];
        let mut indices: Vec<u32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();
        for row in rows.iter_mut() {
            row.sort_unstable();
            for &j in row.iter() {
                indices.push(j);
                data.push(1);
            }
            indptr.push(indices.len() as u32);
        }

        CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (n, n))
    }

    /// Densify a connectivity CSR for readable assertions.
    ///
    /// ### Params
    ///
    /// * `csr` - The abstracted graph.
    ///
    /// ### Returns
    ///
    /// A `k` by `k` dense copy, indexed `[row][col]`.
    fn densify(csr: &CompressedSparseData2<f64>) -> Vec<Vec<f64>> {
        let k = csr.nrows();
        let mut out = vec![vec![0.0; k]; k];
        for (i, row) in out.iter_mut().enumerate() {
            for idx in csr.indptr[i] as usize..csr.indptr[i + 1] as usize {
                row[csr.indices[idx] as usize] = csr.data[idx];
            }
        }
        out
    }

    /// A ring of directed edges over `members`, so every node has out-degree
    /// one and the partition is internally connected.
    ///
    /// ### Params
    ///
    /// * `members` - Node indices in the ring.
    ///
    /// ### Returns
    ///
    /// The directed edges of the ring.
    fn ring(members: &[usize]) -> Vec<(usize, usize)> {
        (0..members.len())
            .map(|i| (members[i], members[(i + 1) % members.len()]))
            .collect()
    }

    #[test]
    fn test_two_blocks_bridged_by_one_edge() {
        // Two rings of 4, one directed edge 3 -> 4 between them.
        let mut edges = ring(&[0, 1, 2, 3]);
        edges.extend(ring(&[4, 5, 6, 7]));
        edges.push((3, 4));
        let graph = directed_csr(8, &edges);
        let partitions = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let res: PartitionConnectivity<f64> =
            partition_connectivities(&graph, &partitions, None).unwrap();

        assert_eq!(res.sizes, vec![4, 4]);

        // e[0] = 5 (four ring edges plus the bridge), e[1] = 4, s = (4, 4),
        // n = 8, so expected = (5 * 4 + 4 * 4) / 7 = 36 / 7 and observed = 1.
        let dense = densify(&res.connectivities);
        let want = 1.0 / (36.0 / 7.0);
        assert_relative_eq!(dense[0][1], want, epsilon = 1e-12);
        assert_relative_eq!(dense[1][0], want, epsilon = 1e-12);
        assert_eq!(dense[0][0], 0.0);
        assert_eq!(dense[1][1], 0.0);
        assert!(want < 1.0);
    }

    #[test]
    fn test_chain_of_three_partitions_leaves_the_ends_unconnected() {
        let mut edges = ring(&[0, 1]);
        edges.extend(ring(&[2, 3]));
        edges.extend(ring(&[4, 5]));
        edges.push((1, 2));
        edges.push((3, 4));
        let graph = directed_csr(6, &edges);
        let partitions = vec![0, 0, 1, 1, 2, 2];

        let res: PartitionConnectivity<f64> =
            partition_connectivities(&graph, &partitions, None).unwrap();

        let dense = densify(&res.connectivities);
        assert!(dense[0][1] > 0.0);
        assert!(dense[1][2] > 0.0);
        assert_eq!(dense[0][2], 0.0);
        assert_eq!(dense[2][0], 0.0);
    }

    #[test]
    fn test_connectivity_is_capped_at_one() {
        // Two singleton partitions joined in both directions. e = (1, 1),
        // s = (1, 1), n = 2, so expected = (1 + 1) / 1 = 2 while observed is
        // also 2. Adding a self loop to node 0 pushes e[0] to 2 and the ratio
        // to 2 / 3 < 1, so keep the pair clean and check the boundary instead.
        let graph = directed_csr(2, &[(0, 1), (1, 0)]);
        let partitions = vec![0, 1];

        let res: PartitionConnectivity<f64> =
            partition_connectivities(&graph, &partitions, None).unwrap();

        let dense = densify(&res.connectivities);
        assert_relative_eq!(dense[0][1], 1.0, epsilon = 1e-12);

        // Now make the observation exceed the null: four nodes, two per
        // partition, fully cross-connected and nothing internal. e = (4, 4),
        // s = (2, 2), n = 4, expected = (8 + 8) / 3 = 16 / 3 = 5.33 while
        // observed is 8, so the raw ratio is 1.5 and must clamp.
        let cross = directed_csr(
            4,
            &[
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (3, 0),
                (3, 1),
            ],
        );
        let res: PartitionConnectivity<f64> =
            partition_connectivities(&cross, &[0, 0, 1, 1], None).unwrap();

        assert_relative_eq!(densify(&res.connectivities)[0][1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_self_loops_only_move_the_expected_value() {
        // A self loop is a within-partition edge: it lands on the diagonal, so
        // it raises e[a] and therefore lowers the connectivity, but it never
        // becomes an entry of its own.
        let plain = directed_csr(2, &[(0, 1), (1, 0)]);
        let looped = directed_csr(2, &[(0, 0), (0, 1), (1, 0)]);

        let a: PartitionConnectivity<f64> =
            partition_connectivities(&plain, &[0, 1], None).unwrap();
        let b: PartitionConnectivity<f64> =
            partition_connectivities(&looped, &[0, 1], None).unwrap();

        let (da, db) = (densify(&a.connectivities), densify(&b.connectivities));
        assert!(db[0][1] < da[0][1]);
        assert_eq!(db[0][0], 0.0);
        assert_eq!(db[1][1], 0.0);
    }

    #[test]
    fn test_single_partition_has_no_connectivities() {
        let graph = directed_csr(4, &ring(&[0, 1, 2, 3]));

        let res: PartitionConnectivity<f64> =
            partition_connectivities(&graph, &[0, 0, 0, 0], None).unwrap();

        assert_eq!(res.sizes, vec![4]);
        assert_eq!(res.connectivities.shape, (1, 1));
        assert_eq!(res.connectivities.get_nnz(), 0);
    }

    #[test]
    fn test_declared_empty_partition_is_kept() {
        let graph = directed_csr(2, &[(0, 1), (1, 0)]);

        let res: PartitionConnectivity<f64> =
            partition_connectivities(&graph, &[0, 1], Some(4)).unwrap();

        assert_eq!(res.sizes, vec![1, 1, 0, 0]);
        assert_eq!(res.connectivities.shape, (4, 4));
        assert_eq!(res.connectivities.get_nnz(), 2);
    }

    #[test]
    fn test_both_counting_arms_agree() {
        // Same graph and labels, once under the parallel limit and once above
        // it, by declaring more partitions than are used. The extra partitions
        // are empty, so the connectivities must be identical.
        let mut edges = ring(&[0, 1, 2, 3]);
        edges.extend(ring(&[4, 5, 6, 7]));
        edges.push((3, 4));
        edges.push((7, 0));
        let graph = directed_csr(8, &edges);
        let partitions = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let small: PartitionConnectivity<f64> =
            partition_connectivities(&graph, &partitions, Some(PARALLEL_PARTITION_LIMIT)).unwrap();
        let large: PartitionConnectivity<f64> =
            partition_connectivities(&graph, &partitions, Some(PARALLEL_PARTITION_LIMIT + 1))
                .unwrap();

        assert_eq!(small.connectivities.data, large.connectivities.data);
        assert_eq!(small.connectivities.indices, large.connectivities.indices);
    }

    #[test]
    fn test_rejects_label_count_mismatch() {
        let graph = directed_csr(3, &[(0, 1)]);

        let res = partition_connectivities::<f64, _>(&graph, &[0, 1], None);

        assert!(matches!(
            res,
            Err(BixverseErrors::CommunityAssignmentMismatch)
        ));
    }

    #[test]
    fn test_rejects_out_of_range_label() {
        let graph = directed_csr(2, &[(0, 1)]);

        let res = partition_connectivities::<f64, _>(&graph, &[0, 5], Some(2));

        assert!(matches!(
            res,
            Err(BixverseErrors::PartitionLabelOutOfRange {
                label: 5,
                n_partitions: 2
            })
        ));
    }

    #[test]
    fn test_rejects_too_many_partitions() {
        let graph = directed_csr(2, &[(0, 1)]);

        let res = partition_connectivities::<f64, _>(&graph, &[0, 1], Some(MAX_PARTITIONS + 1));

        assert!(matches!(res, Err(BixverseErrors::TooManyPartitions { .. })));
    }

    #[test]
    fn test_rejects_csc_input() {
        let graph = directed_csr(2, &[(0, 1)]).transform();

        let res = partition_connectivities::<f64, _>(&graph, &[0, 1], None);

        assert!(matches!(res, Err(BixverseErrors::SparseMatrixMustBeCsr)));
    }
}
