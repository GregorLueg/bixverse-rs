//! Biology-conservation metrics.
//!
//! The counterpart to [`super::batch`]: these score how much of the cell type
//! structure survived an integration. cLISI is not here, because it is
//! [`super::shared::lisi`] on cell type labels followed by
//! [`super::shared::LisiResult::clisi_norm`]; nothing about the calculation
//! changes with the meaning of the labels.

use faer::MatRef;
use rayon::prelude::*;

use super::shared::{SilhouetteResult, dense_labels, silhouette_width, summarise};
use crate::core::math::sparse::coo_to_csr;
use crate::graph::graph_components::induced_components;
use crate::prelude::*;

///////////////////
// Cell type ASW //
///////////////////

/// Cell type average silhouette width, rescaled onto `[0, 1]`
///
/// The same silhouette as [`super::batch::batch_silhouette_width`], but on cell
/// type labels and mapped through `(s + 1) / 2` so that 1 means the types sit
/// in cleanly separated groups and 0 that they are inverted. Note the opposite
/// reading to the batch version, where separation is the failure mode.
///
/// ### Params
///
/// * `embedding` - Low-dimensional embedding, cells x dimensions.
/// * `labels` - Cell type label per cell, one per row of `embedding`.
/// * `subsample` - Optional cap on the number of cells used. The calculation is
///   O(n^2), so pass one for anything above a few tens of thousands of cells.
/// * `seed` - Random seed for the subsampling.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// A [`SilhouetteResult`] whose scores are already rescaled onto `[0, 1]`.
///
/// ### References
///
/// Luecken et al., Nat. Methods, 2022
pub fn cell_type_asw(
    embedding: MatRef<f32>,
    labels: &[usize],
    subsample: Option<usize>,
    seed: usize,
    verbose: bool,
) -> Result<SilhouetteResult, BixverseErrors> {
    let result = silhouette_width(embedding, labels, subsample, seed, verbose)?;

    let per_cell: Vec<f32> = result.per_cell.iter().map(|&s| (s + 1.0) / 2.0).collect();
    let (mean, median) = summarise(&per_cell);

    Ok(SilhouetteResult {
        per_cell,
        mean,
        median,
    })
}

////////////////////////
// Graph connectivity //
////////////////////////

/// Results from a graph connectivity calculation
#[derive(Clone, Debug)]
pub struct GraphConnectivityResult {
    /// Per-label share of that label's cells sitting in its largest connected
    /// component, in `(0, 1]`
    pub per_label: Vec<f32>,
    /// Mean across labels
    pub mean: f32,
    /// Median across labels
    pub median: f32,
}

/// Connectivity of each cell type within the integrated kNN graph
///
/// For every label, the subgraph induced on that label's cells is taken and the
/// share of those cells landing in its largest weakly connected component is
/// recorded. A score of 1 means the cell type forms one connected blob; a low
/// score means integration shredded it into islands, which is the failure both
/// LISI and the silhouette are blind to.
///
/// Edge direction is ignored, so an asymmetric kNN graph does not need to be
/// symmetrised first.
///
/// ### Params
///
/// * `knn_indices` - Neighbour indices per cell.
/// * `labels` - Cell type label per cell, one per row of `knn_indices`.
///
/// ### Returns
///
/// A [`GraphConnectivityResult`] with one score per label plus summaries. The
/// per-label order follows first appearance in `labels`.
///
/// ### References
///
/// Luecken et al., Nat. Methods, 2022
pub fn graph_connectivity(
    knn_indices: &[Vec<usize>],
    labels: &[usize],
) -> Result<GraphConnectivityResult, BixverseErrors> {
    let n = knn_indices.len();
    let (labels, n_labels) = dense_labels(n, labels)?;

    let nnz: usize = knn_indices.iter().map(|x| x.len()).sum();
    let mut rows: Vec<u32> = Vec::with_capacity(nnz);
    let mut cols: Vec<u32> = Vec::with_capacity(nnz);
    for (i, neighbours) in knn_indices.iter().enumerate() {
        for &j in neighbours {
            if j >= n {
                return Err(BixverseErrors::SliceIndexOutOfBounds { index: j, len: n });
            }
            rows.push(i as u32);
            cols.push(j as u32);
        }
    }
    let vals = vec![1.0f32; rows.len()];
    let graph = coo_to_csr(&rows, &cols, &vals, (n, n));

    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_labels];
    for (i, &l) in labels.iter().enumerate() {
        members[l].push(i);
    }

    let per_label: Result<Vec<f32>, BixverseErrors> = members
        .par_iter()
        .map(|member| {
            if member.is_empty() {
                return Ok(0.0);
            }
            let (n_components, component) = induced_components(&graph, member)?;

            let mut sizes = vec![0usize; n_components];
            for &c in &component {
                sizes[c] += 1;
            }
            let largest = sizes.iter().max().copied().unwrap_or(0);

            Ok(largest as f32 / member.len() as f32)
        })
        .collect();
    let per_label = per_label?;

    let (mean, median) = summarise(&per_label);

    Ok(GraphConnectivityResult {
        per_label,
        mean,
        median,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod bio_tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Two tight, far-apart cell types: the rescaled silhouette saturates near
    /// 1, the good end.
    #[test]
    fn test_cell_type_asw_separated_labels_near_one() {
        let data: Vec<f32> = vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 20.0, 20.0, 20.1, 20.0, 20.0, 20.1,
        ];
        let embd = MatRef::<f32>::from_row_major_slice(&data, 6, 2);
        let labels = vec![0, 0, 0, 1, 1, 1];

        let res = cell_type_asw(embd, &labels, None, 42, false).unwrap();

        assert!(res.mean > 0.99, "mean ASW was {}", res.mean);
        assert!(res.per_cell.iter().all(|&s| (0.0..=1.0).contains(&s)));
    }

    /// The rescaling is exactly `(s + 1) / 2` of the raw silhouette.
    #[test]
    fn test_cell_type_asw_is_rescaled_silhouette() {
        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 6.0, 0.0];
        let embd = MatRef::<f32>::from_row_major_slice(&data, 4, 2);
        let labels = vec![0, 0, 1, 1];

        let raw = silhouette_width(embd, &labels, None, 1, false).unwrap();
        let scaled = cell_type_asw(embd, &labels, None, 1, false).unwrap();

        for (r, s) in raw.per_cell.iter().zip(scaled.per_cell.iter()) {
            assert_relative_eq!((r + 1.0) / 2.0, s, epsilon = 1e-6);
        }
    }

    /// A label whose cells form one chain scores 1; the other is deliberately
    /// cut into two equal halves and scores 0.5.
    #[test]
    fn test_graph_connectivity_detects_split_label() {
        // Label 0 on cells 0..4 forms a path; label 1 on cells 4..8 is two
        // disjoint pairs.
        let knn = vec![
            vec![1],
            vec![0, 2],
            vec![1, 3],
            vec![2],
            vec![5],
            vec![4],
            vec![7],
            vec![6],
        ];
        let labels = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let res = graph_connectivity(&knn, &labels).unwrap();

        assert_relative_eq!(res.per_label[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(res.per_label[1], 0.5, epsilon = 1e-6);
        assert_relative_eq!(res.mean, 0.75, epsilon = 1e-6);
    }

    /// Edges leaving the label are ignored, so a label held together only by
    /// cells of another type still scores as fragmented.
    #[test]
    fn test_graph_connectivity_ignores_cross_label_edges() {
        let knn = vec![vec![1], vec![0, 2], vec![1]];
        let labels = vec![0, 1, 0];

        let res = graph_connectivity(&knn, &labels).unwrap();

        assert_relative_eq!(res.per_label[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(res.per_label[1], 1.0, epsilon = 1e-6);
    }

    /// Direction is ignored: a one-way kNN graph must not be scored as
    /// disconnected.
    #[test]
    fn test_graph_connectivity_ignores_edge_direction() {
        let knn = vec![vec![1], vec![], vec![3], vec![]];
        let labels = vec![0, 0, 1, 1];

        let res = graph_connectivity(&knn, &labels).unwrap();

        assert_relative_eq!(res.mean, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_graph_connectivity_rejects_out_of_range_neighbour() {
        let knn = vec![vec![1], vec![9]];
        assert!(matches!(
            graph_connectivity(&knn, &[0, 1]),
            Err(BixverseErrors::SliceIndexOutOfBounds { index: 9, len: 2 })
        ));
    }
}
