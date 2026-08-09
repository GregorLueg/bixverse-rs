//! PAGA trajectory inference.
//!
//! Coarse-grains the cell-level kNN graph over a clustering and scores every
//! pair of clusters against a random-connection null, giving an abstracted
//! graph of a few dozen nodes plus the spanning tree that best explains its
//! topology. The heavy lifting lives in
//! [crate::graph::graph_abstraction::partition_connectivities] and
//! [crate::graph::spanning_tree::maximum_spanning_forest]; this file only turns
//! kNN output into the graph they want.
//!
//! Only the reference's `v1.2` model is implemented. It is scanpy's default and
//! the only one that runs off the raw kNN graph; `v1.0` wants the UMAP fuzzy
//! simplicial set instead. RNA-velocity transitions are likewise not here,
//! there being no velocity anywhere in this crate.
//!
//! The confidence scores read as observed-over-expected connections. The null
//! over-estimates what counts as connected in real data, so the reference
//! deliberately publishes no p-value alongside them and neither does this.
//!
//! ### Deliberate divergences from the reference implementation
//!
//! 1. The tree is the maximum spanning forest of the connectivities rather than
//!    the minimum spanning tree of their reciprocals. Same edges, one pass
//!    fewer, and it never takes `1 / x` on a value that can be arbitrarily
//!    small.
//! 2. `connectivities_tree` is stored symmetrically. scipy emits one direction
//!    per tree edge and the reference keeps that asymmetry, which surprises
//!    anything iterating stored entries.
//! 3. A disconnected abstracted graph gives a spanning *forest* rather than
//!    failing. The reference inherits the same behaviour from scipy but does
//!    not say so.
//! 4. Ties in the tree resolve by `(weight, lo, hi)` instead of by whichever
//!    order scipy's traversal happens to reach them in.
//! 5. Partitions declared but never used are kept at size zero instead of being
//!    dropped, so the output stays aligned with an R factor's levels.
//! 6. Degenerate inputs return typed errors rather than raising out of scipy.
//!
//! ### References
//!
//! Wolf, et al., Genome Biol., 2019. "PAGA: graph abstraction reconciles
//! clustering with trajectory inference through a topology preserving map of
//! single cells."

use crate::graph::graph_abstraction::partition_connectivities;
use crate::graph::spanning_tree::maximum_spanning_forest;
use crate::prelude::*;

/////////////
// Structs //
/////////////

/// Results of a PAGA run.
pub struct PagaResult<T>
where
    T: Clone,
{
    /// Abstracted graph, clusters by clusters, symmetric CSR with a zero
    /// diagonal. Stored values lie in `(0, 1]` and read as confidence in a
    /// connection between the two clusters.
    pub connectivities: CompressedSparseData2<T>,
    /// Maximum spanning forest of `connectivities`, carrying the original
    /// connectivity values on the retained edges. Symmetric CSR of the same
    /// shape. A forest rather than a tree whenever the abstracted graph is
    /// disconnected.
    pub connectivities_tree: CompressedSparseData2<T>,
    /// Cells per cluster, indexed by cluster id.
    pub sizes: Vec<usize>,
}

//////////
// Main //
//////////

/// Run PAGA over a kNN graph and a clustering.
///
/// Takes the kNN indices as they come out of the crate's search backends, self
/// hit already dropped, and reads them as a **directed** graph: an edge runs
/// from cell `i` to every cell in `knn_indices[i]`. That is what the reference
/// does with its distance matrix, and the asymmetry matters, since a cell's
/// out-degree is what anchors the null model.
///
/// Distances are not used. Neither is any expression data, so one
/// implementation serves the disk-backed cell store and in-memory meta cells
/// alike.
///
/// ### Params
///
/// * `knn_indices` - kNN indices per cell, self excluded. Out-of-range and
///   self-referencing entries are dropped rather than counted.
/// * `partitions` - Cluster label per cell, one per entry of `knn_indices`.
///   Typically the output of [crate::graph::community_detections].
/// * `n_partitions` - Declared cluster count. `None` derives it as
///   `max(label) + 1`; pass it to keep empty levels of a factor.
///
/// ### Returns
///
/// The abstracted graph, its spanning forest and the cluster sizes, or an error
/// when the label count does not match the cell count, a label sits outside
/// `n_partitions`, or the cluster count is implausibly large.
///
/// ### References
///
/// Wolf, et al., Genome Biol., 2019.
pub fn run_paga<T>(
    knn_indices: &[Vec<usize>],
    partitions: &[usize],
    n_partitions: Option<usize>,
) -> Result<PagaResult<T>, BixverseErrors>
where
    T: BixverseFloat + BixverseNumeric,
{
    let graph = directed_knn_csr(knn_indices);

    let abstracted = partition_connectivities::<T, u8>(&graph, partitions, n_partitions)?;
    let connectivities_tree = maximum_spanning_forest(&abstracted.connectivities)?;

    Ok(PagaResult {
        connectivities: abstracted.connectivities,
        connectivities_tree,
        sizes: abstracted.sizes,
    })
}

/////////////
// Helpers //
/////////////

/// Build the binarised directed adjacency of a kNN graph.
///
/// Values are `u8` ones and are never read: the coarsening consumes the
/// sparsity pattern alone, so this costs a byte per edge instead of the four a
/// `f32` distance layer would.
///
/// Neither [crate::core::math::sparse::coo_to_csr] nor
/// [crate::single_cell::sc_processing::knn::knn_to_sparse_dist] fits. The first
/// sorts the whole edge list when the rows are already grouped; the second
/// drops zero-distance neighbours, which would silently delete edges between
/// duplicate cells.
///
/// ### Params
///
/// * `knn_indices` - kNN indices per cell. Self hits and out-of-range indices
///   are dropped; the row is then sorted, which CSR wants and the backends do
///   not provide (they return neighbours by distance).
///
/// ### Returns
///
/// A square CSR adjacency over the cells with one entry per surviving
/// neighbour.
fn directed_knn_csr(knn_indices: &[Vec<usize>]) -> CompressedSparseData2<u8> {
    let n = knn_indices.len();

    let mut indptr: Vec<u32> = Vec::with_capacity(n + 1);
    let mut indices: Vec<u32> = Vec::with_capacity(knn_indices.iter().map(|r| r.len()).sum());
    indptr.push(0);

    let mut row: Vec<u32> = Vec::new();
    for (i, neighbours) in knn_indices.iter().enumerate() {
        row.clear();
        row.extend(
            neighbours
                .iter()
                .filter(|&&j| j != i && j < n)
                .map(|&j| j as u32),
        );
        row.sort_unstable();
        row.dedup();
        indices.extend_from_slice(&row);
        indptr.push(indices.len() as u32);
    }

    let data = vec![1u8; indices.len()];

    CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (n, n))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Densify a symmetric CSR for readable assertions.
    ///
    /// ### Params
    ///
    /// * `csr` - The matrix to densify.
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

    /// Three tight blocks of four cells, with 0 the trunk and 1, 2 the arms.
    ///
    /// Every cell lists the other three of its block. The trunk is wired to
    /// both arms; the arms touch only through it.
    ///
    /// ### Returns
    ///
    /// kNN indices for the twelve cells.
    fn y_shape() -> Vec<Vec<usize>> {
        let blocks = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]];
        let mut knn: Vec<Vec<usize>> = vec![Vec::new(); 12];
        for block in blocks.iter() {
            for &c in block.iter() {
                knn[c] = block.iter().copied().filter(|&o| o != c).collect();
            }
        }
        // Trunk to arms, in both directions so the bridge is not one-sided.
        knn[3].push(4);
        knn[4].push(3);
        knn[0].push(8);
        knn[8].push(0);
        knn
    }

    #[test]
    fn test_y_shape_connects_the_trunk_to_both_arms_only() {
        let knn = y_shape();
        let partitions = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

        let res: PagaResult<f64> = run_paga(&knn, &partitions, None).unwrap();

        assert_eq!(res.sizes, vec![4, 4, 4]);

        let dense = densify(&res.connectivities);
        assert!(dense[0][1] > 0.0);
        assert!(dense[0][2] > 0.0);
        assert_eq!(dense[1][2], 0.0);
        // Symmetric, zero diagonal.
        for a in 0..3 {
            assert_eq!(dense[a][a], 0.0);
            for b in 0..3 {
                assert_relative_eq!(dense[a][b], dense[b][a], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_tree_keeps_both_arms_of_the_y() {
        let knn = y_shape();
        let partitions = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

        let res: PagaResult<f64> = run_paga(&knn, &partitions, None).unwrap();

        // Three nodes, two edges, both incident on the trunk.
        assert_eq!(res.connectivities_tree.get_nnz(), 4);
        let tree = densify(&res.connectivities_tree);
        assert!(tree[0][1] > 0.0);
        assert!(tree[0][2] > 0.0);
        assert_eq!(tree[1][2], 0.0);
    }

    #[test]
    fn test_tree_drops_the_weakest_edge_of_a_triangle() {
        // Three blocks of four wired into a triangle, with two mutual bridges
        // on 0-1 and on 0-2 but only one on 1-2. Working it through: es is
        // (16, 15, 15) against sizes of four each over twelve cells, so the
        // expectations are 124/11 twice and 120/11 once, against observed
        // counts of 4, 4 and 2. The 1-2 link is the weakest by a clear margin
        // and is the edge the forest has to give up.
        let blocks = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]];
        let mut knn: Vec<Vec<usize>> = vec![Vec::new(); 12];
        for block in blocks.iter() {
            for &c in block.iter() {
                knn[c] = block.iter().copied().filter(|&o| o != c).collect();
            }
        }
        for (a, b) in [(3, 4), (2, 5), (0, 8), (1, 9), (7, 8)] {
            knn[a].push(b);
            knn[b].push(a);
        }
        let partitions = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

        let res: PagaResult<f64> = run_paga(&knn, &partitions, None).unwrap();

        let dense = densify(&res.connectivities);
        assert_relative_eq!(dense[0][1], 4.0 / (124.0 / 11.0), epsilon = 1e-12);
        assert_relative_eq!(dense[0][2], 4.0 / (124.0 / 11.0), epsilon = 1e-12);
        assert_relative_eq!(dense[1][2], 2.0 / (120.0 / 11.0), epsilon = 1e-12);
        assert!(dense[1][2] < dense[0][1]);

        let tree = densify(&res.connectivities_tree);
        assert_eq!(res.connectivities_tree.get_nnz(), 4);
        assert_relative_eq!(tree[0][1], dense[0][1], epsilon = 1e-12);
        assert_relative_eq!(tree[0][2], dense[0][2], epsilon = 1e-12);
        assert_eq!(tree[1][2], 0.0);
    }

    #[test]
    fn test_self_hits_and_out_of_range_neighbours_are_dropped() {
        // Cell 0 lists itself and a cell that does not exist. Neither may reach
        // the counting, so the result must match the cleaned-up input exactly.
        let dirty = vec![vec![0, 1, 99], vec![0], vec![1]];
        let clean = vec![vec![1], vec![0], vec![1]];

        let a: PagaResult<f64> = run_paga(&dirty, &[0, 1, 1], None).unwrap();
        let b: PagaResult<f64> = run_paga(&clean, &[0, 1, 1], None).unwrap();

        assert_eq!(a.connectivities.data, b.connectivities.data);
        assert_eq!(a.connectivities.indices, b.connectivities.indices);
        assert_eq!(a.sizes, b.sizes);
    }

    #[test]
    fn test_duplicate_neighbours_are_counted_once() {
        let duped = vec![vec![1, 1, 1], vec![0]];
        let clean = vec![vec![1], vec![0]];

        let a: PagaResult<f64> = run_paga(&duped, &[0, 1], None).unwrap();
        let b: PagaResult<f64> = run_paga(&clean, &[0, 1], None).unwrap();

        assert_eq!(a.connectivities.data, b.connectivities.data);
    }

    /// The kNN pattern of a scanpy reference run.
    ///
    /// Twenty-four cells in three unequal, linearly arranged 2-D blobs, put
    /// through `sc.pp.neighbors(n_neighbors=5, use_rep="X")`. These are the
    /// stored column indices of `adata.obsp["distances"]`, row by row, which is
    /// exactly the directed graph `sc.tl.paga` binarises.
    ///
    /// ### Returns
    ///
    /// kNN indices for the twenty-four cells.
    fn scanpy_reference_knn() -> Vec<Vec<usize>> {
        vec![
            vec![1, 8, 5, 2],
            vec![0, 15, 9, 3],
            vec![5, 8, 0, 9],
            vec![11, 9, 14, 16],
            vec![7, 8, 5, 0],
            vec![2, 8, 0, 7],
            vec![7, 5, 8, 2],
            vec![4, 8, 5, 6],
            vec![5, 2, 0, 7],
            vec![3, 1, 2, 0],
            vec![16, 17, 3, 11],
            vec![14, 16, 15, 3],
            vec![17, 16, 14, 11],
            vec![15, 14, 11, 1],
            vec![11, 16, 17, 15],
            vec![11, 14, 13, 1],
            vec![17, 14, 11, 10],
            vec![16, 14, 12, 11],
            vec![22, 20, 12, 19],
            vec![23, 21, 22, 20],
            vec![18, 12, 19, 22],
            vec![19, 23, 22, 18],
            vec![18, 21, 19, 20],
            vec![19, 21, 22, 20],
        ]
    }

    #[test]
    fn test_matches_the_scanpy_reference() {
        // Reference values from `sc.tl.paga(adata, groups="grp")` on the run
        // described by `scanpy_reference_knn`. Both are exact: out-degree is a
        // flat four, so es = (40, 32, 24) against sizes (10, 8, 6) over 24
        // cells, giving expected values of 640/23 and 384/23 against observed
        // counts of 8 and 2.
        let knn = scanpy_reference_knn();
        let partitions: Vec<usize> = (0..24)
            .map(|i| {
                if i < 10 {
                    0
                } else if i < 18 {
                    1
                } else {
                    2
                }
            })
            .collect();

        let res: PagaResult<f64> = run_paga(&knn, &partitions, None).unwrap();

        assert_eq!(res.sizes, vec![10, 8, 6]);

        let dense = densify(&res.connectivities);
        assert_relative_eq!(dense[0][1], 0.287_500_000_000_000_03, epsilon = 1e-15);
        assert_relative_eq!(dense[1][2], 0.119_791_666_666_666_67, epsilon = 1e-15);
        assert_eq!(dense[0][2], 0.0);

        // The reference tree is the same two edges. It stores them one-way;
        // this port stores both directions, hence four entries rather than two.
        let tree = densify(&res.connectivities_tree);
        assert_eq!(res.connectivities_tree.get_nnz(), 4);
        assert_relative_eq!(tree[0][1], dense[0][1], epsilon = 1e-15);
        assert_relative_eq!(tree[1][2], dense[1][2], epsilon = 1e-15);
    }

    #[test]
    fn test_rejects_label_count_mismatch() {
        let knn = vec![vec![1], vec![0]];

        let res = run_paga::<f64>(&knn, &[0], None);

        assert!(matches!(
            res,
            Err(BixverseErrors::CommunityAssignmentMismatch)
        ));
    }

    #[test]
    fn test_rejects_out_of_range_label() {
        let knn = vec![vec![1], vec![0]];

        let res = run_paga::<f64>(&knn, &[0, 7], Some(2));

        assert!(matches!(
            res,
            Err(BixverseErrors::PartitionLabelOutOfRange { .. })
        ));
    }
}
