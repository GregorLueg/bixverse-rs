use faer::{Mat, MatRef};
use rand::prelude::*;

use crate::graph::graph_structures::{adjacency_to_laplacian, get_knn_graph_adj};
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Helper function to get K-means clustering
///
/// ### Params
///
/// * `data` - The data on which to apply k-means clustering
/// * `k` - Number of clusters
/// * `max_iters` - Maximum number of iterations
/// * `seed` - Random seed for reproducibility
///
/// ### Return
///
/// Vector with usizes, indicating cluster membership
fn kmeans<T>(data: &MatRef<T>, k: usize, max_iters: usize, seed: usize) -> Vec<usize>
where
    T: BixverseFloat,
{
    let n = data.nrows();
    let d = data.ncols();
    let mut labels = vec![0; n];
    let mut centroids = Mat::zeros(k, d);

    let mut rng = StdRng::seed_from_u64(seed as u64);
    centroids
        .as_mut()
        .row_mut(0)
        .copy_from(data.row(rng.random_range(0..n)));

    // TODO: update this to use the SIMD optimised code from ann-search-rs

    for i in 1..k {
        let mut distances = vec![T::infinity(); n];
        for j in 0..n {
            for c in 0..i {
                let dist: T = (0..d)
                    .map(|dim| (data[(j, dim)] - centroids[(c, dim)]).powi(2))
                    .fold(T::zero(), |acc, x| acc + x);
                distances[j] = distances[j].min(dist);
            }
        }
        let idx = distances
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        centroids.as_mut().row_mut(i).copy_from(data.row(idx));
    }

    for _ in 0..max_iters {
        for i in 0..n {
            let mut min_dist = T::infinity();
            for j in 0..k {
                let dist: T = (0..d)
                    .map(|dim| (data[(i, dim)] - centroids[(j, dim)]).powi(2))
                    .fold(T::zero(), |acc, x| acc + x);
                if dist < min_dist {
                    min_dist = dist;
                    labels[i] = j;
                }
            }
        }

        let mut counts = vec![0; k];
        let mut new_centroids = Mat::zeros(k, d);
        for i in 0..n {
            counts[labels[i]] += 1;
            for j in 0..d {
                new_centroids[(labels[i], j)] += data[(i, j)];
            }
        }
        for i in 0..k {
            if counts[i] > 0 {
                for j in 0..d {
                    new_centroids[(i, j)] /= T::from_usize(counts[i]).unwrap();
                }
            }
        }
        centroids = new_centroids;
    }

    labels
}

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
    max_iters: usize,
    seed: usize,
) -> Vec<usize>
where
    T: BixverseFloat,
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

    kmeans(&features.as_ref(), n_clusters, max_iters, seed)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

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
        let labels = spectral_clustering(&sim.as_ref(), 1, 2, 100, 42);

        assert_eq!(labels.len(), 4);
        assert_eq!(labels[0], labels[1]); // 0 and 1 are together
        assert_eq!(labels[2], labels[3]); // 2 and 3 are together
        assert_ne!(labels[0], labels[2]); // The blocks are distinct
    }
}
