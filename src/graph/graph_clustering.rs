//! Graph-based clustering algorithms, namely spectral clustering

use ann_search_rs::prelude::AnnSearchFloat;
use faer::{Mat, MatRef};

use super::graph_structures::{adjacency_to_laplacian, get_knn_graph_adj};

use crate::ml::clustering::k_means::*;
use crate::prelude::*;

////////////////////
// Main functions //
////////////////////

/// Spectral clustering
///
/// ### Params
///
/// * `similarities` - The matrix of similarities
/// * `k_neighbours` - Number of neighbours to consider
/// * `n_cluster` - Number of clusters to detect.
/// * `max_iters` - Maximum iterations for the k-means clustering.
/// * `seed` - For reproducibility purposes in the centroid initialisation
///
/// ### Returns
///
/// Vector with usizes, indicating cluster membership
pub fn spectral_clustering<T>(
    similarities: &MatRef<T>,
    k_neighbours: usize,
    n_clusters: usize,
    kmeans_params: Option<KMeansParamsWrappers>,
    seed: usize,
) -> Result<Vec<usize>, BixverseErrors>
where
    T: BixverseFloat + AnnSearchFloat,
{
    let adjacency = get_knn_graph_adj(similarities, k_neighbours);

    let laplacian = adjacency_to_laplacian(&adjacency.as_ref(), true);

    let eigendecomp = laplacian.eigen().unwrap();
    let eigenvalues = eigendecomp.S().column_vector();
    let eigenvectors = eigendecomp.U();

    let mut indices: Vec<usize> = (0..eigenvalues.nrows()).collect();
    indices.sort_by(|&a, &b| eigenvalues[a].re.partial_cmp(&eigenvalues[b].re).unwrap());

    let mut features = Mat::zeros(similarities.nrows(), n_clusters);
    for i in 0..similarities.nrows() {
        for j in 0..n_clusters {
            features[(i, j)] = eigenvectors[(i, indices[j])].re
        }
    }

    for i in 0..features.nrows() {
        let norm: T = (0..n_clusters)
            .map(|j| features[(i, j)].powi(2))
            .fold(T::zero(), |acc, x| acc + x)
            .sqrt();
        if norm > T::from_f64(1e-10).unwrap() {
            for j in 0..n_clusters {
                features[(i, j)] /= norm;
            }
        }
    }

    let (_, assignments) = k_means_clusters(
        features.as_ref(),
        "euclidean",
        n_clusters,
        kmeans_params,
        seed,
        false,
    )?;

    Ok(assignments)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    /// Two similarity blocks with weak cross-noise are recovered as two
    /// clusters.
    #[test]
    fn test_spectral_clustering_block_diagonal() {
        // 4x4 matrix, two distinct blocks: (0,1) and (2,3)
        let mut sim: Mat<f64> = Mat::zeros(4, 4);
        // Block 1
        sim[(0, 0)] = 1.0;
        sim[(0, 1)] = 0.9;
        sim[(1, 0)] = 0.9;
        sim[(1, 1)] = 1.0;
        // Block 2
        sim[(2, 2)] = 1.0;
        sim[(2, 3)] = 0.9;
        sim[(3, 2)] = 0.9;
        sim[(3, 3)] = 1.0;
        // Weak noise between blocks
        sim[(0, 2)] = 0.1;
        sim[(2, 0)] = 0.1;

        // Extract 2 clusters looking at top 1 neighbor
        let labels = spectral_clustering(&sim.as_ref(), 1, 2, None, 42).unwrap();

        assert_eq!(labels.len(), 4);
        assert_eq!(labels[0], labels[1]); // 0 and 1 are together
        assert_eq!(labels[2], labels[3]); // 2 and 3 are together
        assert_ne!(labels[0], labels[2]); // The blocks are distinct
    }
}
