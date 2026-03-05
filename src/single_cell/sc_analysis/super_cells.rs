//! Implementation of the SuperCell approach from Bilous, et al., BMC
//! Bioinform., 2022

use crate::prelude::*;

use crate::graph::community_detections::walktrap_sparse_graph;
use crate::graph::graph_structures::knn_to_sparse_graph;

///////////////
// SuperCell //
///////////////

/// Structure for the SuperCell parameters
///
/// ### Fields
///
/// **SuperCell params**
///
/// * `walk_length` - Walk length for the Walktrap algorithm
/// * `graining_factor` - Graining level of data (proportion of number of single
///   cells in the initial dataset to the number of metacells in the final
///   dataset)
/// * `linkage_dist` - Which type of distance metric to use for the linkage.
///
/// **General kNN params**
///
/// * `knn_params` - All of the kNN parameters
#[derive(Clone, Debug)]
pub struct SuperCellParams {
    /// Walk length for the Walktrap algorithm
    pub walk_length: usize,
    /// Graining level of data (proportion of number of single cells in the
    /// initial dataset to the number of metacells in the final dataset)
    pub graining_factor: f64,
    /// Which type of distance metric to use for the linkage.
    pub linkage_dist: String,
    /// Parameters for the various approximate nearest neighbour searches
    /// in ann-search-rs
    pub knn_params: KnnParams,
}

/// SuperCell algorithm
///
/// ### Params
///
/// * `knn_mat` - The kNN matrix
/// * `walk_length` - Walk length for the Walktrap algorithm
/// * `no_meta_cells` - Number of communities, i.e., metacells to identify
/// * `linkage_dist` - The distance metric to use for the linkage.
/// * `verbose` - Controls the verbosity of the function
///
/// ### Returns
///
/// Membership of included cells to MetaCells.
pub fn supercell(
    knn_mat: &[Vec<usize>],
    walk_length: usize,
    no_meta_cells: usize,
    linkage_dist: &str,
    verbose: bool,
) -> Vec<usize> {
    let knn_graph: SparseGraph<f32> = knn_to_sparse_graph(knn_mat);
    walktrap_sparse_graph(
        &knn_graph,
        walk_length,
        no_meta_cells,
        linkage_dist,
        verbose,
    )
}
