//! Implementation of the SuperCell approach from Bilous, et al., BMC
//! Bioinform., 2022. There is also an additional option to apply a kernel
//! similar to the pre-print of SuperCell 2.0, please refer to Hérault, et al.,
//! bioRxiv, 2026

use ann_search_rs::utils::dist::SimdDistance;
use rustc_hash::FxHashMap;

use crate::core::math::sparse::coo_to_csr;
use crate::graph::community_detections::walktrap_sparse_graph;
use crate::graph::graph_structures::knn_to_sparse_graph;
use crate::prelude::*;

///////////////
// SuperCell //
///////////////

////////////
// Params //
////////////

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
    /// [`crate::prelude::KnnParams`] for the various approximate nearest
    /// neighbour searches in ann-search-rs
    pub knn_params: KnnParams,
    /// Shall a kernel be applied as in SuperCell 2.0, see Hérault, et al.,
    /// bioRxiv, 2026
    pub use_kernel: bool,
    /// Optional choice for k_ith neighbour to use
    pub k_ith: Option<usize>,
}

/////////////
// Helpers //
/////////////

/// Transform kNN to a sparse graph with kernel applied
///
/// ### Params
///
/// * `knn` - The kNN indices without self.
/// * `dist` - The distances to the nearest neighbours.
/// * `squared_dist` - If the distance is squared (for example Euclidean
///   squared).
/// * `kith` - An option for the k_ith neighbour
///
/// ### Returns
///
/// The [SparseGraph] with the kernel applied to the edges.
pub fn knn_to_sparse_graph_kernel<T>(
    knn: &[Vec<usize>],
    dist: &[Vec<T>],
    squared_dist: bool,
    kith: Option<usize>,
) -> SparseGraph<T>
where
    T: BixverseFloat + std::iter::Sum + BixverseNumeric,
{
    let n = knn.len();
    let k = knn.first().map(|v| v.len()).unwrap_or(0);
    let kith = kith.unwrap_or_else(|| (k / 2).saturating_sub(2));

    let sigmas: Vec<T> = dist
        .iter()
        .map(|d| {
            if squared_dist {
                d[kith].sqrt()
            } else {
                d[kith]
            }
        })
        .collect();

    let mut acc: FxHashMap<(usize, usize), (T, usize)> = FxHashMap::default();
    for (i, (nbrs, dists)) in knn.iter().zip(dist.iter()).enumerate() {
        for (&j, &d) in nbrs.iter().zip(dists.iter()) {
            if i == j {
                continue;
            }
            let key = if i < j { (i, j) } else { (j, i) };
            let entry = acc.entry(key).or_insert((T::zero(), 0));
            entry.0 += if squared_dist { d.sqrt() } else { d };
            entry.1 += 1;
        }
    }

    let mut rows = Vec::with_capacity(acc.len() * 2);
    let mut cols = Vec::with_capacity(acc.len() * 2);
    let mut vals = Vec::with_capacity(acc.len() * 2);
    for ((i, j), (sum, count)) in acc {
        let d = sum / T::from_usize(count).unwrap();
        let w = (-(d / sigmas[j]).powi(2)).exp() + (-(d / sigmas[i]).powi(2)).exp();
        rows.push(i);
        cols.push(j);
        vals.push(w);
        rows.push(j);
        cols.push(i);
        vals.push(w);
    }

    let adjacency = coo_to_csr(&rows, &cols, &vals, (n, n));
    SparseGraph::new(n, adjacency, false)
}

//////////
// Main //
//////////

/// SuperCell algorithm
///
/// ### Params
///
/// * `knn_mat` - The kNN neighbour indices
/// * `knn_dist` - The kNN neighbour distances
/// * `params` - The [SuperCellParams] defining if a kernel shall be applied,
///   walk length, etc.
/// * `squared_dist` - If the distance is squared (for example Euclidean
///   squared).
/// * `no_meta_cells` - Number of communities, i.e., metacells to identify
/// * `verbose` - Controls the verbosity of the function
///
/// ### Returns
///
/// Membership of included cells to MetaCells.
pub fn supercell<T>(
    knn_indices: &[Vec<usize>],
    knn_dist: &[Vec<T>],
    params: &SuperCellParams,
    squared_dist: bool,
    no_meta_cells: usize,
    verbose: bool,
) -> Vec<usize>
where
    T: BixverseFloat + std::iter::Sum + BixverseNumeric + SimdDistance,
{
    let knn_graph: SparseGraph<T> = if params.use_kernel {
        if verbose {
            println!("Using the kernel approach like in SuperCell 2.0")
        }
        knn_to_sparse_graph_kernel(knn_indices, knn_dist, squared_dist, params.k_ith)
    } else {
        if verbose {
            println!("Using the original version of SuperCell on the kNN graph")
        }
        knn_to_sparse_graph(knn_indices, true)
    };
    walktrap_sparse_graph(&knn_graph, params.walk_length, no_meta_cells, verbose)
}
