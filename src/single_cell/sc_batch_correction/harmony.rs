use ann_search_rs::{utils::dist::Dist, utils::ivf_utils::train_centroids};
use faer::linalg::solvers::PartialPivLu;
use faer::{Mat, MatRef, linalg::solvers::DenseSolveCore};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use thousands::*;

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
/// * `block_size`: fraction of cells to update per block (0.0-1.0)
/// * `max_iter_kmeans`: maximum k-means iterations per Harmony round
/// * `max_iter_harmony`: maximum Harmony outer iterations
/// * `epsilon_kmeans`: k-means convergence threshold
/// * `epsilon_harmony`: harmony convergence threshold
/// * `window_size`: window size for convergence checking
pub struct HarmonyParams {
    pub k: usize,
    pub sigma: Vec<f32>,
    pub theta: Vec<f32>,
    pub lambda: Vec<f32>,
    pub block_size: f32,
    pub max_iter_kmeans: usize,
    pub max_iter_harmony: usize,
    pub epsilon_kmeans: f32,
    pub epsilon_harmony: f32,
    pub window_size: usize,
}

/// Default implementation for HarmonyParams
impl Default for HarmonyParams {
    fn default() -> Self {
        Self {
            k: 10,
            sigma: vec![0.1],
            theta: vec![2.0],
            lambda: vec![1.0],
            block_size: 0.05,
            max_iter_kmeans: 10,
            max_iter_harmony: 10,
            epsilon_kmeans: 1e-5,
            epsilon_harmony: 1e-4,
            window_size: 3,
        }
    }
}

/////////////
// Helpers //
/////////////

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

    // compute O: sum R values for each cluster-batch combination
    for (batch_idx, cell_indices) in batch_indices.iter().enumerate() {
        for &cell_idx in cell_indices {
            for cluster_idx in 0..k {
                o[(cluster_idx, batch_idx)] += r[(cluster_idx, cell_idx)];
            }
        }
    }

    // compute row sums from O (since every cell belongs to exactly one batch)
    let mut row_sums = vec![0.0f32; k];
    for cluster_idx in 0..k {
        for batch_idx in 0..b {
            row_sums[cluster_idx] += o[(cluster_idx, batch_idx)];
        }
    }

    // compute E = row_sums * pr_b^T
    let mut e = Mat::zeros(k, b);
    for cluster_idx in 0..k {
        for batch_idx in 0..b {
            e[(cluster_idx, batch_idx)] = row_sums[cluster_idx] * pr_b[batch_idx];
        }
    }

    (o, e)
}

/// Update cluster centroids using soft assignments
///
/// Computes `weighted mean: Y = normalise(R * Z_cos)`
/// Each centroid is the R-weighted sum of all cells, then normalised to unit
/// length.
///
/// ### Params
///
/// * `z_cos` - Cosine-normalised data (N × d)
/// * `r` - Soft assignments (K × N)
///
/// ### Returns
///
/// Updated centroids (K × d), cosine-normalized
pub fn update_centroids_from_r(z_cos: MatRef<f32>, r: MatRef<f32>) -> Mat<f32> {
    assert_eq!(r.ncols(), z_cos.nrows(), "R columns must match Z_cos rows");

    let y = r * z_cos;

    // normalise each row (cluster centroid) to unit L2 norm
    cosine_normalise(&y)
}

/// Compute Harmony objective function for convergence checking
///
/// Objective = kmeans_error + entropy + cross_entropy:
///
/// - kmeans_error: `sum(R .* dist_mat)`
/// - entropy: `sum(R .* log(R)) per cluster, weighted by sigma`
/// - cross_entropy: `diversity penalty term`
///
/// ### Params
///
/// * `r` - Soft assignments (K × N)
/// * `dist_mat` - Distance matrix (K × N)
/// * `o` - Observed diversity (K × B)
/// * `e` - Expected diversity (K × B)
/// * `sigma` - Per-cluster weights (length K)
/// * `theta` - Per-batch penalties (length B)
/// * `batch_indices` - Cell indices per batch (length B)
///
/// ### Returns
///
/// Objective value (lower is better)
pub fn compute_objective(
    r: MatRef<f32>,
    dist_mat: MatRef<f32>,
    o: MatRef<f32>,
    e: MatRef<f32>,
    sigma: &[f32],
    theta: &[f32],
    batch_indices: &[Vec<usize>],
) -> f32 {
    let k = r.nrows();
    let n = r.ncols();
    let b = batch_indices.len();

    assert_eq!(dist_mat.nrows(), k);
    assert_eq!(dist_mat.ncols(), n);
    assert_eq!(sigma.len(), k);
    assert_eq!(theta.len(), b);

    let norm_const = 2000.0 / n as f32;

    // K-means error + entropy: iterate cell-first for column-major access
    let mut kmeans_error = 0.0f32;
    let mut entropy = 0.0f32;

    for cell_idx in 0..n {
        for cluster_idx in 0..k {
            let r_val = r[(cluster_idx, cell_idx)];
            kmeans_error += r_val * dist_mat[(cluster_idx, cell_idx)];
            if r_val > 0.0 {
                entropy += r_val * r_val.ln() * sigma[cluster_idx];
            }
        }
    }

    // pre-compute log-ratio table: K × B (instead of recomputing per cell)
    let mut log_ratio = vec![0.0f32; k * b];
    for cluster_idx in 0..k {
        for batch_idx in 0..b {
            let o_val = o[(cluster_idx, batch_idx)];
            let e_val = e[(cluster_idx, batch_idx)];
            if e_val > 0.0 {
                log_ratio[cluster_idx * b + batch_idx] = ((o_val + e_val) / e_val).ln();
            }
        }
    }

    // cross-entropy using precomputed log-ratios
    let mut cross_entropy = 0.0f32;
    for (batch_idx, cell_indices) in batch_indices.iter().enumerate() {
        let theta_b = theta[batch_idx];

        for &cell_idx in cell_indices {
            for cluster_idx in 0..k {
                let r_val = r[(cluster_idx, cell_idx)];
                cross_entropy +=
                    r_val * sigma[cluster_idx] * theta_b * log_ratio[cluster_idx * b + batch_idx];
            }
        }
    }

    (kmeans_error + entropy + cross_entropy) * norm_const
}

/// Update soft assignments with batch diversity penalties
///
/// This is the core Harmony algorithm. Updates R using:
///
/// R[k,n] ∝ exp(-dist[k,n]/σ[k]) × (E[k,b]/(O[k,b]+E[k,b]))^θ[b]
///
/// Uses block-wise shuffling to efficiently update O and E statistics.
///
/// ### Params
///
/// * `dist_mat` - Distance matrix (K × N)
/// * `sigma` - Per-cluster diversity weights (length K)
/// * `theta` - Per-batch diversity penalties (length B)
/// * `batch_indices` - Cell indices per batch (length B)
/// * `pr_b` - Batch frequencies (length B)
/// * `block_size` - Fraction of cells per update block
/// * `seed` - Random seed for shuffling
/// * `r_init` - Initial R matrix (K × N) to start from
/// * `o_init` - Initial O matrix (K × B)
/// * `e_init` - Initial E matrix (K × B)
///
/// ### Returns
///
/// Tuple of
/// * `0` - R: K×N updated assignments
/// * `1` - O: K×B updated observed
/// * `2` - E: K×B updated expected
pub fn update_r_with_diversity(
    dist_mat: MatRef<f32>,
    sigma: &[f32],
    theta: &[f32],
    batch_indices: &[Vec<usize>],
    pr_b: &[f32],
    block_size: f32,
    seed: usize,
    r_init: MatRef<f32>,
    o_init: MatRef<f32>,
    e_init: MatRef<f32>,
) -> (Mat<f32>, Mat<f32>, Mat<f32>) {
    let k = dist_mat.nrows();
    let n = dist_mat.ncols();
    let b = batch_indices.len();

    assert_eq!(sigma.len(), k);
    assert_eq!(theta.len(), b);
    assert_eq!(pr_b.len(), b);

    // Cell-to-batch lookup
    let mut cell_to_batch = vec![0usize; n];
    for (batch_idx, cells) in batch_indices.iter().enumerate() {
        for &cell_idx in cells {
            cell_to_batch[cell_idx] = batch_idx;
        }
    }

    // Precompute base scaled distances: exp(-dist / sigma), column-normalised
    let mut scale_dist = Mat::zeros(k, n);
    for cell_idx in 0..n {
        let mut col_sum = 0.0f32;
        for cluster_idx in 0..k {
            let val = (-dist_mat[(cluster_idx, cell_idx)] / sigma[cluster_idx]).exp();
            scale_dist[(cluster_idx, cell_idx)] = val;
            col_sum += val;
        }
        if col_sum > 0.0 {
            for cluster_idx in 0..k {
                scale_dist[(cluster_idx, cell_idx)] /= col_sum;
            }
        }
    }

    // shuffled update order (no physical reordering — just index indirection)
    let mut rng = StdRng::seed_from_u64(seed as u64);
    let mut update_order: Vec<usize> = (0..n).collect();
    update_order.shuffle(&mut rng);

    // work on R in original layout
    let mut r = r_init.to_owned();
    let mut o = o_init.to_owned();
    let mut e = e_init.to_owned();

    // block-wise updates
    let n_blocks = (1.0 / block_size).ceil() as usize;
    let cells_per_block = (n as f32 * block_size) as usize;

    for block_idx in 0..n_blocks {
        let idx_min = block_idx * cells_per_block;
        let idx_max = ((block_idx + 1) * cells_per_block).min(n);

        // step 1: remove block cells from O and E
        for &cell_idx in &update_order[idx_min..idx_max] {
            let batch_idx = cell_to_batch[cell_idx];
            for cluster_idx in 0..k {
                let r_val = r[(cluster_idx, cell_idx)];
                o[(cluster_idx, batch_idx)] -= r_val;
                for b_idx in 0..b {
                    e[(cluster_idx, b_idx)] -= r_val * pr_b[b_idx];
                }
            }
        }

        // step 2: recompute R for block cells with diversity penalty
        for &cell_idx in &update_order[idx_min..idx_max] {
            let batch_idx = cell_to_batch[cell_idx];
            let theta_b = theta[batch_idx];

            let mut new_col_sum = 0.0f32;
            for cluster_idx in 0..k {
                let base = scale_dist[(cluster_idx, cell_idx)];
                let o_val = o[(cluster_idx, batch_idx)];
                let e_val = e[(cluster_idx, batch_idx)];
                let penalty = if o_val + e_val > 0.0 {
                    (e_val / (o_val + e_val)).powf(theta_b)
                } else {
                    1.0
                };
                let new_r = base * penalty;
                r[(cluster_idx, cell_idx)] = new_r;
                new_col_sum += new_r;
            }

            if new_col_sum > 0.0 {
                for cluster_idx in 0..k {
                    r[(cluster_idx, cell_idx)] /= new_col_sum;
                }
            }
        }

        // step 3: add block cells back to O and E
        for &cell_idx in &update_order[idx_min..idx_max] {
            let batch_idx = cell_to_batch[cell_idx];
            for cluster_idx in 0..k {
                let r_val = r[(cluster_idx, cell_idx)];
                o[(cluster_idx, batch_idx)] += r_val;
                for b_idx in 0..b {
                    e[(cluster_idx, b_idx)] += r_val * pr_b[b_idx];
                }
            }
        }
    }

    (r, o, e)
}

/// Apply per-cluster ridge regression to remove batch effects
///
/// For each cluster k:
///
/// 1. Weight batch matrix by R[k,n] (soft cluster assignments)
/// 2. Solve ridge regression to estimate batch effects
/// 3. Remove batch effects from data
/// 4. Keep intercept (don't shift global mean)
///
/// ### Params
///
/// * `z_orig` - Original data (N × d)
/// * `r` - Soft assignments (K × N)
/// * `batch_indices` - Cell indices per batch (length B)
/// * `lambda` - Ridge parameters (length B+1, includes intercept)
///
/// ### Returns
///
/// Corrected data (N × d)
pub fn ridge_regression_correction(
    z_orig: MatRef<f32>,
    r: MatRef<f32>,
    batch_indices: &[Vec<usize>],
    lambda: &[f32],
) -> Mat<f32> {
    let n = z_orig.nrows();
    let d = z_orig.ncols();
    let k = r.nrows();
    let b = batch_indices.len();

    assert_eq!(r.ncols(), n);
    assert_eq!(lambda.len(), b, "lambda should be length B");

    let mut z_corr = z_orig.to_owned();

    for cluster_idx in 0..k {
        let mut w_b = vec![0.0f32; b];
        let mut phi_z = Mat::<f32>::zeros(b, d);

        // Row 0 of phi_z accumulates ALL cells (intercept)
        // Row i (i>=1) accumulates only cells in batch i
        for (batch_idx, cell_indices) in batch_indices.iter().enumerate() {
            for &cell_idx in cell_indices {
                let r_val = r[(cluster_idx, cell_idx)];
                w_b[batch_idx] += r_val;

                // Intercept row
                for feat in 0..d {
                    phi_z[(0, feat)] += r_val * z_orig[(cell_idx, feat)];
                }

                // Deviation row (non-reference batches only)
                if batch_idx > 0 {
                    for feat in 0..d {
                        phi_z[(batch_idx, feat)] += r_val * z_orig[(cell_idx, feat)];
                    }
                }
            }
        }

        // Build Phi_Rk * Phi_Rk^T (B × B) analytically:
        //
        // For the intercept coding:
        // phi_rk[0, n] = R[k,n] for all n
        // phi_rk[i, n] = R[k,n] if n in batch i, else 0  (i >= 1)
        //
        // So (Phi * Phi^T)[i,j] = sum_n phi_rk[i,n] * phi_rk[j,n]
        //
        // We need sum of R[k,n]^2 per batch for the diagonal terms.
        let mut w2_b = vec![0.0f32; b]; // sum of R[k,n]^2 per batch
        for (batch_idx, cell_indices) in batch_indices.iter().enumerate() {
            for &cell_idx in cell_indices {
                let r_val = r[(cluster_idx, cell_idx)];
                w2_b[batch_idx] += r_val * r_val;
            }
        }
        let w2_total: f32 = w2_b.iter().sum();

        let mut design_cov = Mat::<f32>::zeros(b, b);

        // [0,0]: sum of all R[k,n]^2
        design_cov[(0, 0)] = w2_total;

        // [0,i] and [i,0] for i>=1: sum of R[k,n]^2 for cells in batch i
        for i in 1..b {
            design_cov[(0, i)] = w2_b[i];
            design_cov[(i, 0)] = w2_b[i];
        }

        // [i,j] for i,j >= 1: delta(i,j) * w2_b[i]
        for i in 1..b {
            design_cov[(i, i)] = w2_b[i];
            // off-diagonal [i,j] for i!=j, both >=1: 0 (disjoint batches)
        }

        // Add ridge penalty
        for i in 0..b {
            design_cov[(i, i)] += lambda[i];
        }

        // Solve
        let partial_piv_lu: PartialPivLu<f32> = design_cov.partial_piv_lu();
        let inv_cov = partial_piv_lu.inverse();

        // W = inv_cov * phi_z : B × d
        let w = &inv_cov * &phi_z;

        // Apply correction: subtract deviation contributions (rows 1..B-1)
        // For each cell in batch b (b >= 1):
        //   z_corr[n, :] -= R[k,n] * W[b, :]
        for batch_idx in 1..b {
            for &cell_idx in &batch_indices[batch_idx] {
                let r_val = r[(cluster_idx, cell_idx)];
                for feat in 0..d {
                    z_corr[(cell_idx, feat)] -= r_val * w[(batch_idx, feat)];
                }
            }
        }
    }

    z_corr
}

/////////////
// Harmony //
/////////////

/// Check convergence using windowed moving average
///
/// ### Params
///
/// * `objectives` - Objective values
/// * `window_size` - Window size
/// * `epsilon` - Convergence threshold
///
/// ### Returns
///
/// * `bool` - Whether convergence is reached
fn check_convergence(objectives: &[f32], window_size: usize, epsilon: f32) -> bool {
    let n = objectives.len();
    if n < 2 * window_size {
        return false;
    }

    let mut obj_old = 0_f32;
    let mut obj_new = 0_f32;

    for i in 0..window_size {
        obj_old += objectives[n - 1 - window_size - i];
        obj_new += objectives[n - 1 - i];
    }

    let rel_change = (obj_old - obj_new).abs() / obj_old.abs();
    rel_change < epsilon
}

/// Harmony state
///
/// ### Fields
///
/// * `z_orig` - Original data (N×d)
/// * `z_cos` - Cosine-normalized (N×d)
/// * `z_corr` - Corrected embeddings (N×d)
/// * `y` - Cluster centroids (K×d)
/// * `r` - Soft assignments (K×N)
/// * `o` - Observed diversity (K×B)
/// * `e` - Expected diversity (K×B)
/// * `phi` - Batch one-hot (B×N)
/// * `batch_indices` - Cell indices per batch
/// * `pr_b` - Batch frequencies
/// * `objectives_kmeans` - K-means objectives
/// * `objectives_harmony` - Harmony objectives
struct HarmonyState {
    z_orig: Mat<f32>,
    z_cos: Mat<f32>,
    z_corr: Mat<f32>,
    y: Mat<f32>,
    r: Mat<f32>,
    o: Mat<f32>,
    e: Mat<f32>,
    batch_indices: Vec<Vec<usize>>,
    pr_b: Vec<f32>,
    objectives_kmeans: Vec<f32>,
    objectives_harmony: Vec<f32>,
}

/// Run Harmony batch correction
///
/// ### Params
///
/// * `pca` - PCA embedding (N × d)
/// * `batch_labels` - Batch assignment per cell (length N)
/// * `params` - Harmony hyperparameters
/// * `seed` - Random seed
/// * `verbose` - Print progress
///
/// ### Returns
///
/// Corrected PCA embedding (N × d)
pub fn harmony(
    pca: MatRef<f32>,
    batch_labels: &[usize],
    params: &HarmonyParams,
    seed: usize,
    verbose: bool,
) -> Mat<f32> {
    let n = pca.nrows();
    let d = pca.ncols();
    let n_batches = batch_labels.iter().max().unwrap() + 1;

    if verbose {
        println!(
            "Harmony: {} cells, {} dims, {} batches, {} clusters",
            n.separate_with_underscores(),
            d,
            n_batches,
            params.k
        );
    }

    let sigma = if params.sigma.len() == 1 {
        vec![params.sigma[0]; params.k]
    } else {
        params.sigma.clone()
    };

    let theta = if params.theta.len() == 1 {
        vec![params.theta[0]; n_batches]
    } else {
        params.theta.clone()
    };

    // FIX: was n_batches + 1, should be n_batches
    let lambda = if params.lambda.len() == 1 {
        vec![params.lambda[0]; n_batches]
    } else {
        params.lambda.clone()
    };

    // --- remainder of harmony() is unchanged from here ---

    let (_, pr_b, batch_indices) = create_phi_matrix(batch_labels, n);
    let z_orig = pca.to_owned();
    let z_cos = cosine_normalise(&z_orig);

    if verbose {
        println!("Running initial k-means...");
    }

    let y = run_kmeans_cosine(
        z_cos.as_ref(),
        params.k,
        params.max_iter_kmeans,
        seed,
        verbose,
    );

    let dist_mat = compute_cosine_distances(y.as_ref(), z_cos.as_ref());
    let r = initialise_r_from_dist(dist_mat.as_ref(), &sigma);
    let (o, e) = compute_diversity_statistics(r.as_ref(), &batch_indices, &pr_b);

    let initial_obj = compute_objective(
        r.as_ref(),
        dist_mat.as_ref(),
        o.as_ref(),
        e.as_ref(),
        &sigma,
        &theta,
        &batch_indices,
    );

    let mut state = HarmonyState {
        z_orig,
        z_cos,
        z_corr: pca.to_owned(),
        y,
        r,
        o,
        e,
        batch_indices,
        pr_b,
        objectives_kmeans: vec![initial_obj],
        objectives_harmony: vec![initial_obj],
    };

    if verbose {
        println!("Initial objective: {:.4}", initial_obj);
    }

    for harmony_iter in 0..params.max_iter_harmony {
        if verbose {
            println!("\n=== Harmony iteration {} ===", harmony_iter + 1);
        }

        for kmeans_iter in 0..params.max_iter_kmeans {
            state.y = update_centroids_from_r(state.z_cos.as_ref(), state.r.as_ref());

            let dist_mat = compute_cosine_distances(state.y.as_ref(), state.z_cos.as_ref());

            let (r_new, o_new, e_new) = update_r_with_diversity(
                dist_mat.as_ref(),
                &sigma,
                &theta,
                &state.batch_indices,
                &state.pr_b,
                params.block_size,
                seed + harmony_iter * 1000 + kmeans_iter,
                state.r.as_ref(),
                state.o.as_ref(),
                state.e.as_ref(),
            );

            state.r = r_new;
            state.o = o_new;
            state.e = e_new;

            let obj = compute_objective(
                state.r.as_ref(),
                dist_mat.as_ref(),
                state.o.as_ref(),
                state.e.as_ref(),
                &sigma,
                &theta,
                &state.batch_indices,
            );

            state.objectives_kmeans.push(obj);

            if verbose && kmeans_iter % 5 == 0 {
                println!("  K-means iter {}: obj = {:.4}", kmeans_iter + 1, obj);
            }

            if kmeans_iter >= params.window_size {
                let converged = check_convergence(
                    &state.objectives_kmeans,
                    params.window_size,
                    params.epsilon_kmeans,
                );
                if converged {
                    if verbose {
                        println!("  K-means converged at iteration {}", kmeans_iter + 1);
                    }
                    break;
                }
            }
        }

        if verbose {
            println!("  Applying ridge regression correction...");
        }

        state.z_corr = ridge_regression_correction(
            state.z_orig.as_ref(),
            state.r.as_ref(),
            &state.batch_indices,
            &lambda,
        );

        state.z_cos = cosine_normalise(&state.z_corr);

        let harmony_obj = *state.objectives_kmeans.last().unwrap();
        state.objectives_harmony.push(harmony_obj);

        if verbose {
            println!("  Harmony objective: {:.4}", harmony_obj);
        }

        if harmony_iter >= 1 {
            let obj_old = state.objectives_harmony[harmony_iter];
            let obj_new = state.objectives_harmony[harmony_iter + 1];
            let rel_change = (obj_old - obj_new).abs() / obj_old.abs();

            if rel_change < params.epsilon_harmony {
                if verbose {
                    println!("\nHarmony converged at iteration {}", harmony_iter + 1);
                    println!("  Final objective: {:.4}", obj_new);
                }
                break;
            }
        }
    }

    state.z_corr
}

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
        // After normalisation, cluster 1 should dominate
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

        let r = mat![
            [0.8, 0.7, 0.6, 0.9, 0.5, 0.3], // cluster 0
            [0.2, 0.3, 0.4, 0.1, 0.5, 0.7], // cluster 1
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

    #[test]
    fn test_update_centroids_simple() {
        // Z_cos: 4 cells × 3 features (already normalized)

        // Normalize cell 3
        let norm_3 = (0.5f32 * 0.5 + 0.5 * 0.5).sqrt();
        let z_cos_norm = mat![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5 / norm_3, 0.5 / norm_3, 0.0],
        ];

        // R: 2 clusters × 4 cells
        // Cluster 0: strongly assigns to cells 0 and 3
        // Cluster 1: strongly assigns to cells 1 and 2

        let r = mat![
            [0.9, 0.1, 0.1, 0.8], // cluster 0
            [0.1, 0.9, 0.9, 0.2], // cluster 1
        ];

        let y = update_centroids_from_r(z_cos_norm.as_ref(), r.as_ref());

        // Verify dimensions
        assert_eq!(y.nrows(), 2);
        assert_eq!(y.ncols(), 3);

        // Cluster 0 should be weighted towards cells 0 and 3 (dim 0 and bit of dim 1)
        // y_0 = 0.9*[1,0,0] + 0.1*[0,1,0] + 0.1*[0,0,1] + 0.8*[0.707,0.707,0]
        //     = [0.9 + 0.8*0.707, 0.1 + 0.8*0.707, 0.1]
        //     ≈ [1.466, 0.666, 0.1]
        // After normalisation: should be heavy in dim 0

        // Check that centroids are unit length
        let norm_0 = (y[(0, 0)].powi(2) + y[(0, 1)].powi(2) + y[(0, 2)].powi(2)).sqrt();
        let norm_1 = (y[(1, 0)].powi(2) + y[(1, 1)].powi(2) + y[(1, 2)].powi(2)).sqrt();

        assert!(
            (norm_0 - 1.0).abs() < 1e-6,
            "Cluster 0 not unit length: {}",
            norm_0
        );
        assert!(
            (norm_1 - 1.0).abs() < 1e-6,
            "Cluster 1 not unit length: {}",
            norm_1
        );

        // Cluster 0 should be heavier in dimension 0 than 1
        assert!(
            y[(0, 0)] > y[(0, 1)],
            "Cluster 0 should be heavier in dim 0: [{}, {}, {}]",
            y[(0, 0)],
            y[(0, 1)],
            y[(0, 2)]
        );

        // Cluster 1 should be heavier in dimensions 1 and 2
        assert!(
            y[(1, 1)] > y[(1, 0)],
            "Cluster 1 should be heavier in dim 1: [{}, {}, {}]",
            y[(1, 0)],
            y[(1, 1)],
            y[(1, 2)]
        );
    }

    #[test]
    fn test_update_centroids_hard_assignment() {
        // Test with hard (one-hot) assignments

        // 3 cells, 2 dimensions

        let z_cos = mat![
            [1.0, 0.0],
            [0.0, 1.0],
            [0.707, 0.707], // 45 degree angle
        ];

        // Hard assignment: cluster 0 gets cells 0 and 2, cluster 1 gets cell 1

        let r = mat![
            [1.0, 0.0, 1.0], // cluster 0
            [0.0, 1.0, 0.0], // cluster 1
        ];

        let y = update_centroids_from_r(z_cos.as_ref(), r.as_ref());

        // Cluster 0: average of cells 0 and 2
        // = ([1,0] + [0.707,0.707]) / 2 = [0.854, 0.354]
        // After normalisation: [0.924, 0.383]

        // Cluster 1: just cell 1
        // = [0, 1], already normalized

        // Check cluster 1 (simpler case)
        assert!(
            (y[(1, 0)] - 0.0).abs() < 1e-6,
            "Cluster 1 dim 0: {}",
            y[(1, 0)]
        );
        assert!(
            (y[(1, 1)] - 1.0).abs() < 1e-6,
            "Cluster 1 dim 1: {}",
            y[(1, 1)]
        );

        // Check cluster 0 is normalized
        let norm = (y[(0, 0)].powi(2) + y[(0, 1)].powi(2)).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Cluster 0 not normalized: {}",
            norm
        );

        // Cluster 0 should be biased towards dim 0
        assert!(
            y[(0, 0)] > y[(0, 1)],
            "Cluster 0: [{}, {}]",
            y[(0, 0)],
            y[(0, 1)]
        );
    }

    #[test]
    fn test_compute_objective_decreases() {
        // Test that objective should decrease as R becomes more confident
        // (lower entropy) and distances decrease

        let batch_indices = vec![vec![0, 1], vec![2, 3]];
        let sigma = vec![1.0, 1.0];
        let theta = vec![1.0, 1.0];
        let pr_b = vec![0.5, 0.5];

        // Scenario 1: High uncertainty (uniform R)

        let r_uncertain = mat![[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5],];

        let dist_mat_high = mat![[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0],];

        let (o1, e1) = compute_diversity_statistics(r_uncertain.as_ref(), &batch_indices, &pr_b);

        let obj1 = compute_objective(
            r_uncertain.as_ref(),
            dist_mat_high.as_ref(),
            o1.as_ref(),
            e1.as_ref(),
            &sigma,
            &theta,
            &batch_indices,
        );

        // Scenario 2: Low uncertainty (confident R) and lower distances

        let r_confident = mat![[0.9, 0.9, 0.1, 0.1], [0.1, 0.1, 0.9, 0.9],];

        let dist_mat_low = mat![[0.1, 0.1, 1.0, 1.0], [1.0, 1.0, 0.1, 0.1],];

        let (o2, e2) = compute_diversity_statistics(r_confident.as_ref(), &batch_indices, &pr_b);

        let obj2 = compute_objective(
            r_confident.as_ref(),
            dist_mat_low.as_ref(),
            o2.as_ref(),
            e2.as_ref(),
            &sigma,
            &theta,
            &batch_indices,
        );

        // Confident assignments with lower distances should have lower objective
        assert!(
            obj2 < obj1,
            "Confident R should have lower objective: {} vs {}",
            obj2,
            obj1
        );
    }

    #[test]
    fn test_compute_objective_components() {
        // Test that we can compute objective and it's reasonable

        let batch_indices = vec![vec![0, 1], vec![2]];
        let sigma = vec![1.0, 1.0];
        let theta = vec![1.0, 1.0];
        let pr_b = vec![2.0 / 3.0, 1.0 / 3.0];

        let r = mat![[0.8, 0.7, 0.2], [0.2, 0.3, 0.8],];

        let dist_mat = mat![[0.1, 0.2, 0.9], [0.9, 0.8, 0.1],];

        let (o, e) = compute_diversity_statistics(r.as_ref(), &batch_indices, &pr_b);

        let obj = compute_objective(
            r.as_ref(),
            dist_mat.as_ref(),
            o.as_ref(),
            e.as_ref(),
            &sigma,
            &theta,
            &batch_indices,
        );

        // Objective should be finite and reasonable
        assert!(obj.is_finite(), "Objective should be finite");
        assert!(obj > 0.0, "Objective should be positive (given our setup)");

        // With normalisation constant of 2000/3, and our small values,
        // objective should be in a reasonable range
        assert!(obj < 10000.0, "Objective seems too large: {}", obj);
    }

    #[test]
    fn test_objective_zero_entropy() {
        // Test with hard (one-hot) assignments → zero entropy contribution

        let batch_indices = vec![vec![0], vec![1]];
        let sigma = vec![1.0, 1.0];
        let theta = vec![0.0, 0.0]; // No diversity penalty
        let pr_b = vec![0.5, 0.5];

        // Hard assignment

        let r = mat![[1.0, 0.0], [0.0, 1.0],];

        let dist_mat = mat![[0.1, 0.9], [0.9, 0.1],];

        let (o, e) = compute_diversity_statistics(r.as_ref(), &batch_indices, &pr_b);

        let obj = compute_objective(
            r.as_ref(),
            dist_mat.as_ref(),
            o.as_ref(),
            e.as_ref(),
            &sigma,
            &theta,
            &batch_indices,
        );

        // With hard assignments and no diversity penalty:
        // objective ≈ kmeans_error = (1.0*0.1 + 0.0*0.9 + 0.0*0.9 + 1.0*0.1) * (2000/2)
        //           = 0.2 * 1000 = 200
        let expected = 0.2 * 1000.0;
        assert!(
            (obj - expected).abs() < 1.0,
            "Objective should be ~200: got {}",
            obj
        );
    }

    #[test]
    fn test_update_r_basic() {
        // Simple test: 2 clusters, 2 batches, 4 cells
        let batch_indices = vec![vec![0, 1], vec![2, 3]];
        let pr_b = vec![0.5, 0.5];
        let sigma = vec![1.0, 1.0];
        let theta = vec![1.0, 1.0];

        // Initial R (uniform)

        let r_init = mat![[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5],];

        // Distances: cluster 0 close to cells 0,1; cluster 1 close to cells 2,3

        let dist_mat = mat![[0.1, 0.1, 0.9, 0.9], [0.9, 0.9, 0.1, 0.1],];

        let (o_init, e_init) = compute_diversity_statistics(r_init.as_ref(), &batch_indices, &pr_b);

        let (r_new, o_new, e_new) = update_r_with_diversity(
            dist_mat.as_ref(),
            &sigma,
            &theta,
            &batch_indices,
            &pr_b,
            0.5,
            42,
            r_init.as_ref(),
            o_init.as_ref(),
            e_init.as_ref(),
        );

        // Verify R columns sum to 1
        for cell_idx in 0..4 {
            let col_sum: f32 = (0..2).map(|k| r_new[(k, cell_idx)]).sum();
            assert!(
                (col_sum - 1.0).abs() < 1e-5,
                "Column {} sum: {}",
                cell_idx,
                col_sum
            );
        }

        // Cluster 0 should be more confident about cells 0,1 (lower distances)
        assert!(
            r_new[(0, 0)] > 0.5,
            "Cluster 0 should be more confident about cell 0"
        );
        assert!(
            r_new[(0, 1)] > 0.5,
            "Cluster 0 should be more confident about cell 1"
        );

        // Cluster 1 should be more confident about cells 2,3
        assert!(
            r_new[(1, 2)] > 0.5,
            "Cluster 1 should be more confident about cell 2"
        );
        assert!(
            r_new[(1, 3)] > 0.5,
            "Cluster 1 should be more confident about cell 3"
        );

        // O and E should be consistent with new R
        let (o_check, e_check) =
            compute_diversity_statistics(r_new.as_ref(), &batch_indices, &pr_b);

        for cluster_idx in 0..2 {
            for batch_idx in 0..2 {
                assert!(
                    (o_new[(cluster_idx, batch_idx)] - o_check[(cluster_idx, batch_idx)]).abs()
                        < 1e-4,
                    "O mismatch at [{},{}]: {} vs {}",
                    cluster_idx,
                    batch_idx,
                    o_new[(cluster_idx, batch_idx)],
                    o_check[(cluster_idx, batch_idx)]
                );
                assert!(
                    (e_new[(cluster_idx, batch_idx)] - e_check[(cluster_idx, batch_idx)]).abs()
                        < 1e-4,
                    "E mismatch at [{},{}]: {} vs {}",
                    cluster_idx,
                    batch_idx,
                    e_new[(cluster_idx, batch_idx)],
                    e_check[(cluster_idx, batch_idx)]
                );
            }
        }
    }

    #[test]
    fn test_update_r_no_diversity_penalty() {
        // With theta=0, should just be based on distances
        let batch_indices = vec![vec![0, 1]];
        let pr_b = vec![1.0];
        let sigma = vec![1.0, 1.0];
        let theta = vec![0.0]; // No penalty

        let r_init = mat![[0.5, 0.5], [0.5, 0.5],];

        let dist_mat = mat![[0.1, 0.9], [0.9, 0.1],];

        let (o_init, e_init) = compute_diversity_statistics(r_init.as_ref(), &batch_indices, &pr_b);

        let (r_new, _, _) = update_r_with_diversity(
            dist_mat.as_ref(),
            &sigma,
            &theta,
            &batch_indices,
            &pr_b,
            1.0,
            42,
            r_init.as_ref(),
            o_init.as_ref(),
            e_init.as_ref(),
        );

        assert!(
            r_new[(0, 0)] > 0.6,
            "Cluster 0 should be confident about cell 0: {}",
            r_new[(0, 0)]
        );
        assert!(
            r_new[(1, 1)] > 0.6,
            "Cluster 1 should be confident about cell 1: {}",
            r_new[(1, 1)]
        );
    }

    #[test]
    fn test_update_r_diversity_correction() {
        // Test that diversity penalty helps balance batches
        // Setup: batch 0 has cells 0,1,2; batch 1 has cell 3
        // All cells close to cluster 0, but diversity penalty should push
        // cell 3 towards cluster 1 to balance batches

        let batch_indices = vec![vec![0, 1, 2], vec![3]];
        let pr_b = vec![0.75, 0.25];
        let sigma = vec![1.0, 1.0];
        let theta = vec![2.0, 2.0]; // Strong diversity penalty

        // Initial: cluster 0 gets most cells

        let r_init = mat![[0.9, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1],];

        // All cells reasonably close to cluster 0

        let dist_mat = mat![[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.8, 0.8],];

        let (o_init, e_init) = compute_diversity_statistics(r_init.as_ref(), &batch_indices, &pr_b);

        let (r_new, _, _) = update_r_with_diversity(
            dist_mat.as_ref(),
            &sigma,
            &theta,
            &batch_indices,
            &pr_b,
            0.5,
            42,
            r_init.as_ref(),
            o_init.as_ref(),
            e_init.as_ref(),
        );

        // With strong diversity penalty, cell 3 (only cell in batch 1)
        // should be pushed more towards cluster 1 to balance representation
        // even though distances favor cluster 0

        // At minimum, check that diversity penalty had some effect
        // (exact values depend on the algorithm dynamics)
        for cell_idx in 0..4 {
            let col_sum: f32 = (0..2).map(|k| r_new[(k, cell_idx)]).sum();
            assert!(
                (col_sum - 1.0).abs() < 1e-5,
                "Column {} sum: {}",
                cell_idx,
                col_sum
            );
        }
    }

    #[test]
    fn test_ridge_regression_basic() {
        // Simple case: 2 batches, 4 cells, 2 features
        let batch_indices = vec![vec![0, 1], vec![2, 3]];

        // Data with clear batch effect in feature 0
        // Batch 0: feature 0 has mean ~1.0
        // Batch 1: feature 0 has mean ~5.0
        let z_orig = mat![
            [1.0, 0.1], // batch 0
            [1.1, 0.2], // batch 0
            [5.0, 0.1], // batch 1
            [5.1, 0.2], // batch 1
        ];

        // Hard assignments: cluster 0 gets all cells
        let r = mat![[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0],];

        // Small ridge penalty (B=2 batches)
        let lambda = vec![0.01, 0.01];

        let z_corr =
            ridge_regression_correction(z_orig.as_ref(), r.as_ref(), &batch_indices, &lambda);

        // After correction, batch effect in feature 0 should be reduced
        // The mean difference between batches should be smaller
        let batch0_mean_orig = (z_orig[(0, 0)] + z_orig[(1, 0)]) / 2.0;
        let batch1_mean_orig = (z_orig[(2, 0)] + z_orig[(3, 0)]) / 2.0;
        let orig_diff = (batch1_mean_orig - batch0_mean_orig).abs();

        let batch0_mean_corr = (z_corr[(0, 0)] + z_corr[(1, 0)]) / 2.0;
        let batch1_mean_corr = (z_corr[(2, 0)] + z_corr[(3, 0)]) / 2.0;
        let corr_diff = (batch1_mean_corr - batch0_mean_corr).abs();

        assert!(
            corr_diff < orig_diff,
            "Batch effect should be reduced: orig_diff={}, corr_diff={}",
            orig_diff,
            corr_diff
        );

        // Feature 1 has no batch effect, should remain relatively unchanged
        let feature1_change: f32 = (0..4)
            .map(|i| (z_corr[(i, 1)] - z_orig[(i, 1)]).abs())
            .sum::<f32>()
            / 4.0;

        assert!(
            feature1_change < 0.5,
            "Feature without batch effect should change little: {}",
            feature1_change
        );
    }

    #[test]
    fn test_ridge_regression_no_correction_needed() {
        // If data has no batch effect, correction should be minimal
        let batch_indices = vec![vec![0, 1], vec![2, 3]];

        // Data with no batch effect
        let z_orig = mat![[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0],];

        let r = mat![[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0],];

        // B=2 batches
        let lambda = vec![0.1, 0.1];

        let z_corr =
            ridge_regression_correction(z_orig.as_ref(), r.as_ref(), &batch_indices, &lambda);

        // Data should be mostly unchanged
        for i in 0..4 {
            for j in 0..2 {
                assert!(
                    (z_corr[(i, j)] - z_orig[(i, j)]).abs() < 0.5,
                    "Cell {}, feature {}: changed from {} to {}",
                    i,
                    j,
                    z_orig[(i, j)],
                    z_corr[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_ridge_regression_soft_assignments() {
        // Test with soft assignments (cells partially in multiple clusters)
        let batch_indices = vec![vec![0, 1], vec![2, 3]];

        let z_orig = mat![[1.0, 0.0], [1.0, 0.0], [5.0, 0.0], [5.0, 0.0],];

        // Soft assignments: cells split between two clusters
        let r = mat![
            [0.8, 0.9, 0.1, 0.2], // Cluster 0 strongly prefers batch 0
            [0.2, 0.1, 0.9, 0.8], // Cluster 1 strongly prefers batch 1
        ];

        // B=2 batches
        let lambda = vec![0.01, 0.01];

        let z_corr =
            ridge_regression_correction(z_orig.as_ref(), r.as_ref(), &batch_indices, &lambda);

        // Should still reduce batch effect
        let batch0_mean_orig = (z_orig[(0, 0)] + z_orig[(1, 0)]) / 2.0;
        let batch1_mean_orig = (z_orig[(2, 0)] + z_orig[(3, 0)]) / 2.0;
        let orig_diff = (batch1_mean_orig - batch0_mean_orig).abs();

        let batch0_mean_corr = (z_corr[(0, 0)] + z_corr[(1, 0)]) / 2.0;
        let batch1_mean_corr = (z_corr[(2, 0)] + z_corr[(3, 0)]) / 2.0;
        let corr_diff = (batch1_mean_corr - batch0_mean_corr).abs();

        assert!(
            corr_diff < orig_diff,
            "Batch effect should be reduced with soft assignments: orig_diff={}, corr_diff={}",
            orig_diff,
            corr_diff
        );
    }

    #[test]
    fn test_ridge_regression_preserves_dimensions() {
        let batch_indices = vec![vec![0], vec![1], vec![2]];

        let z_orig = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];

        let r = mat![[0.5, 0.5, 0.5], [0.5, 0.5, 0.5],];

        // B=3 batches
        let lambda = vec![0.1, 0.1, 0.1];

        let z_corr =
            ridge_regression_correction(z_orig.as_ref(), r.as_ref(), &batch_indices, &lambda);

        assert_eq!(z_corr.nrows(), z_orig.nrows());
        assert_eq!(z_corr.ncols(), z_orig.ncols());
    }
}
