use ann_search_rs::{utils::dist::Dist, utils::ivf_utils::train_centroids};
use faer::{Mat, MatRef};
use rayon::prelude::*;

use crate::prelude::*;
use crate::single_cell::sc_batch_correction::batch_utils::cosine_normalise;
use crate::utils::matrix_utils::{flat_row_major_to_mat, mat_to_flat_row_major};

////////////
// Params //
////////////

/// Parameters for Harmony batch correction.
///
/// ### Fields
///
/// * `k`: number of clusters
/// * `sigma`: cluster diversity weights
/// * `theta`: batch diversity penalties
/// * `lambda`: ridge parameters (None = auto-estimate)
/// * `alpha`: lambda estimation coefficient
/// * `max_iter_harmony`: outer iterations
/// * `max_iter_cluster`: inner k-means iterations
/// * `block_size`: fraction of cells per update block
/// * `epsilon_cluster`: k-means convergence
/// * `epsilon_harmony`: harmony convergence
pub struct HarmonyParams {
    pub k: usize,
    pub sigma: Vec<f32>,
    pub theta: Vec<f32>,
    pub lambda: Option<Vec<f32>>,
    pub alpha: f32,
    pub max_iter_harmony: usize,
    pub max_iter_cluster: usize,
    pub block_size: f32,
    pub epsilon_cluster: f32,
    pub epsilon_harmony: f32,
}

/////////////
// Helpers //
/////////////

/// Harmony state
///
/// ### Fields
///
/// * `z_cos`: cosine-normalized (Nxd)
/// * `z_corr`: corrected embeddings (Nxd)
/// * `y`: cluster centroids (d×K)
/// * `r`: soft assignments (K×N)
/// * `o`: observed diversity (K×B)
/// * `e`: expected diversity (K×B)
/// * `phi`: one-hot batch matrix (B×N)
/// * `objectives`: convergence tracking
struct HarmonyState {
    z_cos: Mat<f32>,
    z_corr: Mat<f32>,
    y: Mat<f32>,
    r: Mat<f32>,
    o: Mat<f32>,
    e: Mat<f32>,
    phi: CompressedSparseData<f32>,
    objectives: Vec<f32>,
}

/// Create one-hot encoded batch matrix Phi (B × N)
///
/// Creates a sparse CSR matrix where Phi[b, n] = 1.0 if cell n belongs to
/// batch b, else 0.0. Each column has exactly one non-zero entry.
///
/// ### Params
///
/// * `batch_labels` - Batch assignment for each cell (length N)
/// * `n_cells` - Number of cells
///
/// ### Returns
///
/// Tuple of (Phi matrix in CSR, batch frequencies, batch indices)
/// - Phi: B × N sparse matrix
/// - Pr_b: Batch frequencies (length B)
/// - batch_indices: Vector of cell indices per batch (length B)
pub fn create_phi_matrix(
    batch_labels: &[usize],
    n_cells: usize,
) -> (CompressedSparseData<f32>, Vec<f32>, Vec<Vec<usize>>) {
    // find number of batches
    let n_batches = batch_labels.iter().max().map(|&x| x + 1).unwrap_or(0);

    // count cells per batch
    let mut batch_counts = vec![0usize; n_batches];
    for &batch in batch_labels {
        batch_counts[batch] += 1;
    }

    // build batch indices (ported from the C++ index structure)
    let mut batch_indices: Vec<Vec<usize>> = vec![Vec::new(); n_batches];
    for (cell_idx, &batch) in batch_labels.iter().enumerate() {
        batch_indices[batch].push(cell_idx);
    }

    // build CSR: each row is a batch, columns are cells
    let mut indptr = Vec::with_capacity(n_batches + 1);
    let mut indices = Vec::with_capacity(n_cells);
    let mut data = Vec::with_capacity(n_cells);

    indptr.push(0);

    for batch in 0..n_batches {
        for &cell_idx in &batch_indices[batch] {
            indices.push(cell_idx);
            data.push(1_f32);
        }
        indptr.push(indices.len());
    }

    let phi = CompressedSparseData {
        data,
        indices,
        indptr,
        cs_type: CompressedSparseFormat::Csr,
        data_2: None,
        shape: (n_batches, n_cells),
    };

    // compute batch frequencies (Pr_b)
    let pr_b: Vec<f32> = batch_counts
        .iter()
        .map(|&count| count as f32 / n_cells as f32)
        .collect();

    (phi, pr_b, batch_indices)
}

/// Run k-means clustering on cosine-normalised data
///
/// Integrates with ann_search_rs k-means implementation which expects flat
/// row-major layout for cache efficiency.
///
/// ### Params
///
/// * `data_cos` - Cosine-normalised data (N × d)
/// * `k` - Number of clusters
/// * `max_iter` - Maximum k-means iterations
/// * `seed` - Random seed
/// * `verbose` - Print progress
///
/// ### Returns
///
/// Cluster centroids (K × d), cosine-normalised
pub fn run_kmeans_cosine(
    data_cos: MatRef<f32>,
    k: usize,
    max_iter: usize,
    seed: usize,
    verbose: bool,
) -> Mat<f32> {
    let n = data_cos.nrows();
    let d = data_cos.ncols();

    if verbose {
        println!("Running k-means: {} cells, {} dims, {} clusters", n, d, k);
    }

    // convert to flat layout
    let data_flat = mat_to_flat_row_major(data_cos);

    // run k-means (returns flat centroids)
    let centroids_flat =
        train_centroids(&data_flat, d, n, k, &Dist::Cosine, max_iter, seed, verbose);

    // convert back to matrix (K × d)
    let centroids = flat_row_major_to_mat(&centroids_flat, k, d);

    // cosine normalise the centroids
    cosine_normalise(&centroids)
}

/// Compute cosine distances between centroids and data
///
/// For cosine-normalised vectors, uses the efficient formula:
/// dist = 2 * (1 - dot_product)
///
/// ### Params
///
/// * `centroids` - Cluster centroids (K × d), must be cosine-normalised
/// * `data_cos` - Data matrix (N × d), must be cosine-normalised
///
/// ### Returns
///
/// Distance matrix (K × N) where dist[k, n] = cosine distance from cluster k
/// to cell n
pub fn compute_cosine_distances(
    centroids: MatRef<f32>, // K × d
    data_cos: MatRef<f32>,  // N × d
) -> Mat<f32> {
    let k = centroids.nrows();
    let n = data_cos.nrows();

    // Compute dot products: centroids × data^T = K×d × d×N = K×N
    let dot_products = centroids * data_cos.transpose();

    // dist = 2 * (1 - dot_product)
    Mat::from_fn(k, n, |i, j| 2.0 * (1.0 - dot_products[(i, j)]))
}

/// Initialise soft cluster assignments from distances
///
/// Converts distances to probabilistic cluster assignments using exponential
/// decay weighted by per-cluster sigma values. Each cell's assignments are
/// normalised to sum to 1.
///
/// ### Params
///
/// * `dist_mat` - Distance matrix (K × N): dist_mat[k, n] = distance from
///   cluster k to cell n
/// * `sigma` - Per-cluster diversity weights (length K)
///
/// ### Returns
///
/// Soft assignment matrix R (K × N) where R[k, n] = probability that cell n
/// belongs to cluster k. Each column sums to 1.
pub fn initialise_r_from_dist(dist_mat: MatRef<f32>, sigma: &[f32]) -> Mat<f32> {
    let k = dist_mat.nrows();
    let n = dist_mat.ncols();
    assert_eq!(sigma.len(), k, "sigma length must match number of clusters");

    let columns: Vec<_> = (0..n)
        .into_par_iter()
        .map(|cell_idx| {
            let mut col_sum = 0.0f32;
            let mut col = vec![0.0f32; k];

            // compute exp(-dist/sigma)
            for cluster_idx in 0..k {
                let dist = dist_mat[(cluster_idx, cell_idx)];
                let val = (-dist / sigma[cluster_idx]).exp();
                col[cluster_idx] = val;
                col_sum += val;
            }

            // normalise column
            for cluster_idx in 0..k {
                col[cluster_idx] /= col_sum;
            }

            col
        })
        .collect();

    Mat::from_fn(k, n, |i, j| columns[j][i])
}

/// Compute observed and expected diversity statistics
///
/// O[k,b] = sum of R[k,n] for all cells n in batch b (observed)
/// E[k,b] = R_k_total * Pr_b[b] (expected under null)
///
/// ### Params
///
/// * `r` - Soft assignments (K × N)
/// * `batch_indices` - Cell indices per batch (length B)
/// * `pr_b` - Batch frequencies (length B)
///
/// ### Returns
///
/// Tuple of (O: K×B matrix, E: K×B matrix)
pub fn compute_diversity_statistics(
    r: MatRef<f32>,
    batch_indices: &[Vec<usize>],
    pr_b: &[f32],
) -> (Mat<f32>, Mat<f32>) {
    let k = r.nrows();
    let b = batch_indices.len();

    assert_eq!(pr_b.len(), b, "pr_b length must match number of batches");

    let mut o = Mat::zeros(k, b);

    // Compute O: sum R values for each cluster-batch combination
    for (batch_idx, cell_indices) in batch_indices.iter().enumerate() {
        for &cell_idx in cell_indices {
            for cluster_idx in 0..k {
                o[(cluster_idx, batch_idx)] += r[(cluster_idx, cell_idx)];
            }
        }
    }

    // Compute row sums from O (since every cell belongs to exactly one batch)
    let mut row_sums = vec![0.0f32; k];
    for cluster_idx in 0..k {
        for batch_idx in 0..b {
            row_sums[cluster_idx] += o[(cluster_idx, batch_idx)];
        }
    }

    // Compute E = row_sums * pr_b^T
    let mut e = Mat::zeros(k, b);
    for cluster_idx in 0..k {
        for batch_idx in 0..b {
            e[(cluster_idx, batch_idx)] = row_sums[cluster_idx] * pr_b[batch_idx];
        }
    }

    (o, e)
}

/////////////
// Harmony //
/////////////

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::mat;

    #[test]
    fn test_create_phi_matrix() {
        let batch_labels = vec![0, 0, 1, 1, 2, 0];
        let n_cells = 6;

        let (phi, pr_b, batch_indices) = create_phi_matrix(&batch_labels, n_cells);

        // Check shape
        assert_eq!(phi.shape, (3, 6));

        // Check batch frequencies
        assert_eq!(pr_b.len(), 3);
        assert!((pr_b[0] - 0.5).abs() < 1e-6); // 3/6
        assert!((pr_b[1] - 0.333333).abs() < 1e-4); // 2/6
        assert!((pr_b[2] - 0.166666).abs() < 1e-4); // 1/6

        // Check batch indices
        assert_eq!(batch_indices[0], vec![0, 1, 5]);
        assert_eq!(batch_indices[1], vec![2, 3]);
        assert_eq!(batch_indices[2], vec![4]);

        // Check CSR structure
        assert_eq!(phi.indptr, vec![0, 3, 5, 6]);
        assert_eq!(phi.indices, vec![0, 1, 5, 2, 3, 4]);
        assert_eq!(phi.data, vec![1.0; 6]);
    }

    #[test]
    fn test_create_phi_single_batch() {
        let batch_labels = vec![0, 0, 0];
        let (phi, pr_b, batch_indices) = create_phi_matrix(&batch_labels, 3);

        assert_eq!(phi.shape, (1, 3));
        assert_eq!(pr_b, vec![1.0]);
        assert_eq!(batch_indices[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_run_kmeans_cosine_basic() {
        // Create two well-separated clusters
        let mut data = Vec::new();

        // Cluster 0: near [1, 0, 0]
        for _ in 0..10 {
            data.push(vec![0.9, 0.1, 0.0]);
        }

        // Cluster 1: near [0, 1, 0]
        for _ in 0..10 {
            data.push(vec![0.1, 0.9, 0.0]);
        }

        let mat = Mat::from_fn(20, 3, |i, j| data[i][j]);
        let mat_cos = cosine_normalise(&mat);

        let centroids = run_kmeans_cosine(mat_cos.as_ref(), 2, 25, 42, false);

        assert_eq!(centroids.nrows(), 2);
        assert_eq!(centroids.ncols(), 3);

        // Check centroids are normalized
        for k in 0..2 {
            let norm: f32 = (0..3)
                .map(|j| centroids[(k, j)].powi(2))
                .sum::<f32>()
                .sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_compute_cosine_distances_identical() {
        // Identical normalised vectors should have distance 0
        let centroids = Mat::from_fn(2, 3, |i, j| {
            match i {
                0 => 1.0 / 3.0f32.sqrt(), // [1/√3, 1/√3, 1/√3]
                _ => {
                    if j == 0 {
                        1.0
                    } else {
                        0.0
                    }
                } // [1, 0, 0]
            }
        });
        let data = centroids.clone();
        let dist = compute_cosine_distances(centroids.as_ref(), data.as_ref());
        // Diagonal should be near zero
        for k in 0..2 {
            assert_relative_eq!(dist[(k, k)], 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_compute_cosine_distances_orthogonal() {
        // Orthogonal normalised vectors should have distance 2
        let centroids = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let data = centroids.clone();

        let dist = compute_cosine_distances(centroids.as_ref(), data.as_ref());

        // Off-diagonal should be 2 (dot product = 0, so 2*(1-0) = 2)
        assert_relative_eq!(dist[(0, 1)], 2.0, epsilon = 1e-5);
        assert_relative_eq!(dist[(1, 0)], 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_initialize_r_from_distances() {
        // Simple 2 clusters, 3 cells
        let dist_data = vec![
            0.0, 2.0, 4.0, // cluster 0 distances
            4.0, 2.0, 0.0, // cluster 1 distances
        ];
        let dist_mat = Mat::from_fn(2, 3, |i, j| dist_data[i * 3 + j]);
        let sigma = vec![1.0, 1.0];

        let r = initialise_r_from_dist(dist_mat.as_ref(), &sigma);

        // Check shape
        assert_eq!(r.nrows(), 2);
        assert_eq!(r.ncols(), 3);

        // Check columns sum to 1
        for col in 0..3 {
            let col_sum: f32 = (0..2).map(|row| r[(row, col)]).sum();
            assert_relative_eq!(col_sum, 1.0, epsilon = 1e-6);
        }

        // Cell 0: close to cluster 0 (dist=0) far from cluster 1 (dist=4)
        assert!(r[(0, 0)] > 0.9);
        assert!(r[(1, 0)] < 0.1);

        // Cell 2: close to cluster 1 (dist=0) far from cluster 0 (dist=4)
        assert!(r[(0, 2)] < 0.1);
        assert!(r[(1, 2)] > 0.9);

        // Cell 1: equidistant (dist=2 to both)
        assert_relative_eq!(r[(0, 1)], 0.5, epsilon = 1e-6);
        assert_relative_eq!(r[(1, 1)], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_initialize_r_different_sigmas() {
        let dist_data = vec![
            1.0, 1.0, // cluster 0
            1.0, 1.0, // cluster 1
        ];
        let dist_mat = Mat::from_fn(2, 2, |i, j| dist_data[i * 2 + j]);

        // cluster 0 has smaller sigma (sharper assignments)
        let sigma = vec![0.5, 2.0];

        let r = initialise_r_from_dist(dist_mat.as_ref(), &sigma);

        // With same distances but different sigmas:
        // exp(-1/0.5) = exp(-2) ≈ 0.135
        // exp(-1/2.0) = exp(-0.5) ≈ 0.606
        // After normalization, cluster 1 should dominate
        assert!(r[(1, 0)] > r[(0, 0)]);
    }

    #[test]
    fn test_compute_diversity_statistics_simple() {
        // Setup: 2 clusters, 3 batches, 6 cells
        // Batch 0: cells 0, 1 (2 cells)
        // Batch 1: cells 2, 3, 4 (3 cells)
        // Batch 2: cell 5 (1 cell)

        let batch_indices = vec![
            vec![0, 1],    // batch 0
            vec![2, 3, 4], // batch 1
            vec![5],       // batch 2
        ];

        let pr_b = vec![2.0 / 6.0, 3.0 / 6.0, 1.0 / 6.0];

        // R matrix (2 clusters × 6 cells)
        // Each column sums to 1.0
        #[rustfmt::skip]
            let r = mat![
                [0.8, 0.7, 0.6, 0.9, 0.5, 0.3],  // cluster 0
                [0.2, 0.3, 0.4, 0.1, 0.5, 0.7],  // cluster 1
            ];

        let (o, e) = compute_diversity_statistics(r.as_ref(), &batch_indices, &pr_b);

        // Verify O dimensions
        assert_eq!(o.nrows(), 2);
        assert_eq!(o.ncols(), 3);

        // Manually compute expected O values
        // O[0, 0] = R[0,0] + R[0,1] = 0.8 + 0.7 = 1.5
        // O[0, 1] = R[0,2] + R[0,3] + R[0,4] = 0.6 + 0.9 + 0.5 = 2.0
        // O[0, 2] = R[0,5] = 0.3
        // O[1, 0] = R[1,0] + R[1,1] = 0.2 + 0.3 = 0.5
        // O[1, 1] = R[1,2] + R[1,3] + R[1,4] = 0.4 + 0.1 + 0.5 = 1.0
        // O[1, 2] = R[1,5] = 0.7

        assert!((o[(0, 0)] - 1.5).abs() < 1e-6, "O[0,0] = {}", o[(0, 0)]);
        assert!((o[(0, 1)] - 2.0).abs() < 1e-6, "O[0,1] = {}", o[(0, 1)]);
        assert!((o[(0, 2)] - 0.3).abs() < 1e-6, "O[0,2] = {}", o[(0, 2)]);
        assert!((o[(1, 0)] - 0.5).abs() < 1e-6, "O[1,0] = {}", o[(1, 0)]);
        assert!((o[(1, 1)] - 1.0).abs() < 1e-6, "O[1,1] = {}", o[(1, 1)]);
        assert!((o[(1, 2)] - 0.7).abs() < 1e-6, "O[1,2] = {}", o[(1, 2)]);

        // Verify E dimensions
        assert_eq!(e.nrows(), 2);
        assert_eq!(e.ncols(), 3);

        // Row sums (total assignment per cluster)
        // row_sum[0] = 0.8 + 0.7 + 0.6 + 0.9 + 0.5 + 0.3 = 3.8
        // row_sum[1] = 0.2 + 0.3 + 0.4 + 0.1 + 0.5 + 0.7 = 2.2

        let expected_row_sum_0 = 3.8;
        let expected_row_sum_1 = 2.2;

        // E[k,b] = row_sum[k] * pr_b[b]
        let expected_e_0_0 = expected_row_sum_0 * pr_b[0];
        let expected_e_0_1 = expected_row_sum_0 * pr_b[1];
        let expected_e_0_2 = expected_row_sum_0 * pr_b[2];
        let expected_e_1_0 = expected_row_sum_1 * pr_b[0];
        let expected_e_1_1 = expected_row_sum_1 * pr_b[1];
        let expected_e_1_2 = expected_row_sum_1 * pr_b[2];

        assert!(
            (e[(0, 0)] - expected_e_0_0).abs() < 1e-6,
            "E[0,0] = {}",
            e[(0, 0)]
        );
        assert!(
            (e[(0, 1)] - expected_e_0_1).abs() < 1e-6,
            "E[0,1] = {}",
            e[(0, 1)]
        );
        assert!(
            (e[(0, 2)] - expected_e_0_2).abs() < 1e-6,
            "E[0,2] = {}",
            e[(0, 2)]
        );
        assert!(
            (e[(1, 0)] - expected_e_1_0).abs() < 1e-6,
            "E[1,0] = {}",
            e[(1, 0)]
        );
        assert!(
            (e[(1, 1)] - expected_e_1_1).abs() < 1e-6,
            "E[1,1] = {}",
            e[(1, 1)]
        );
        assert!(
            (e[(1, 2)] - expected_e_1_2).abs() < 1e-6,
            "E[1,2] = {}",
            e[(1, 2)]
        );
    }

    #[test]
    fn test_diversity_statistics_properties() {
        // Property test: sum of E across batches should equal row sums of R
        // Property test: sum of O across batches should equal row sums of R

        let batch_indices = vec![vec![0, 1, 2], vec![3, 4], vec![5, 6, 7, 8]];

        let pr_b = vec![3.0 / 9.0, 2.0 / 9.0, 4.0 / 9.0];

        // Random-ish R matrix (3 clusters × 9 cells)
        #[rustfmt::skip]
            let r = mat![
                [0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.5, 0.6],
                [0.3, 0.2, 0.4, 0.2, 0.5, 0.1, 0.6, 0.3, 0.3],
                [0.2, 0.2, 0.2, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1],
            ];

        let (o, e) = compute_diversity_statistics(r.as_ref(), &batch_indices, &pr_b);

        // Property 1: Sum of O across batches equals row sum of R
        for cluster_idx in 0..3 {
            let o_sum: f32 = (0..3).map(|b| o[(cluster_idx, b)]).sum();
            let r_row_sum: f32 = (0..9).map(|n| r[(cluster_idx, n)]).sum();
            assert!(
                (o_sum - r_row_sum).abs() < 1e-5,
                "Cluster {}: O sum ({}) != R row sum ({})",
                cluster_idx,
                o_sum,
                r_row_sum
            );
        }

        // Property 2: Sum of E across batches equals row sum of R
        for cluster_idx in 0..3 {
            let e_sum: f32 = (0..3).map(|b| e[(cluster_idx, b)]).sum();
            let r_row_sum: f32 = (0..9).map(|n| r[(cluster_idx, n)]).sum();
            assert!(
                (e_sum - r_row_sum).abs() < 1e-5,
                "Cluster {}: E sum ({}) != R row sum ({})",
                cluster_idx,
                e_sum,
                r_row_sum
            );
        }

        // Property 3: E should be proportional to pr_b
        for cluster_idx in 0..3 {
            let r_row_sum: f32 = (0..9).map(|n| r[(cluster_idx, n)]).sum();
            for batch_idx in 0..3 {
                let expected = r_row_sum * pr_b[batch_idx];
                assert!(
                    (e[(cluster_idx, batch_idx)] - expected).abs() < 1e-5,
                    "E[{},{}] = {}, expected {}",
                    cluster_idx,
                    batch_idx,
                    e[(cluster_idx, batch_idx)],
                    expected
                );
            }
        }
    }
}
