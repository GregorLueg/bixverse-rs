//! Implementation of the SEACells from Persad, et al., Nat. Biotechnol., 2023.
//! Uses some optimised fused kernels to accelerate the FW iterations with
//! better cache locality and reduced memory pressure.

use faer::{Mat, MatRef};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::time::Instant;
use thousands::Separable;

use crate::core::math::sparse::*;
use crate::prelude::*;

////////////////////////
// kNN symmetrisation //
////////////////////////

/// kNN symmetrisation method
#[derive(Clone, Copy, Default)]
pub enum KnnSymmetrisation {
    /// Only intersecting nearest neigbhbours will be considered
    #[default]
    Intersection,
    /// The union of nearest neighbours will be considered
    Union,
}

/// Helper function to parse the SEACell graph generation
///
/// ### Params
///
/// * `s` - Type of graph to build
///
/// ### Returns
///
/// Option of the SeaCellGraphGen
pub fn parse_knn_symmetrisation(s: &str) -> Option<KnnSymmetrisation> {
    match s.to_lowercase().as_str() {
        "intersection" => Some(KnnSymmetrisation::Intersection),
        "union" => Some(KnnSymmetrisation::Union),
        _ => None,
    }
}

////////////
// Params //
////////////

/// Structure to store the SEACells parameters
#[derive(Clone, Debug)]
pub struct SEACellsParams {
    // -- sea cells --
    /// Number of sea cells to detect
    pub n_sea_cells: usize,
    /// Maximum iterations for the Franke-Wolfe algorithm per matrix update.
    pub max_fw_iters: usize,
    /// Defines the convergence threshold. Algorithm stops when
    /// `RSS change < epsilon * RSS(0)`
    pub convergence_epsilon: f32,
    /// Maximum iterations to run SEACells for
    pub max_iter: usize,
    /// Minimum iterations to run SEACells for
    pub min_iter: usize,
    /// Maximum number of cells, before defaulting to a more rapid random
    /// selection of archetypes initially
    pub greedy_threshold: usize,
    /// Which type of KNN graph symmetrisation to use
    pub graph_building: String,
    /// Shall tiny values during the Franke Wolfe updates be pruned.
    /// This can affect numerical stability, but makes runs on large data sets
    /// feasible.
    pub pruning: bool,
    /// Pruning threshold to apply
    pub pruning_threshold: f32,
    /// Optional number of landmarks. If provided, it will use the Nystroem
    /// approach during archetype generation.
    pub n_landmarks: Option<usize>,
    // -- knn --
    /// [KnnParams] for the various approximate nearest neighbour searches
    /// in ann-search-rs
    pub knn_params: KnnParams,
}

/////////////
// Helpers //
/////////////

/// Convert SEACells hard assignments to metacell format
///
/// Transforms flat assignment vector (cell -> SEACell) into grouped format
/// (SEACell -> `[cells]`) suitable for aggregation functions.
///
/// ### Params
///
/// * `assignments` - Vector where `assignments[cell_id] = seacell_id`
/// * `k` - Number of SEACells
///
/// ### Returns
///
/// Vector of vectors, where `result[seacell_id]` contains all cells assigned to
/// that SEACell
pub fn assignments_to_metacells(assignments: &[usize], k: usize) -> Vec<Vec<usize>> {
    let mut metacells = vec![Vec::new(); k];

    for (cell_id, &seacell_id) in assignments.iter().enumerate() {
        metacells[seacell_id].push(cell_id);
    }

    metacells
}

/// Helper function to prune tiny values and renormalise with L1
///
/// ### Params
///
/// * `mat` - Mutable reference to the CompressedSparseData2 to be pruned
/// * `threshold` - Pruning threshold
///
/// ### Returns
///
/// Pruned matrix.
fn prune_and_renormalise(mat: &mut CompressedSparseData2<f32>, threshold: f32) {
    // remove values below threshold
    let mut new_data = Vec::new();
    let mut new_indices = Vec::new();
    let mut new_indptr = vec![0];

    for row in 0..mat.shape.0 {
        let start = mat.indptr[row];
        let end = mat.indptr[row + 1];

        for idx in start..end {
            if mat.data[idx].abs() > threshold {
                new_data.push(mat.data[idx]);
                new_indices.push(mat.indices[idx]);
            }
        }
        new_indptr.push(new_data.len());
    }

    mat.data = new_data;
    mat.indices = new_indices;
    mat.indptr = new_indptr;

    // renormalise columns to maintain sum-to-1 constraint
    normalise_csr_columns_l1(mat);
}

/// Compute the trace (sum of diagonal elements) of a sparse matrix
///
/// ### Params
///
/// * `mat` - Sparse CSR matrix
///
/// ### Returns
///
/// Sum of diagonal elements `mat[i, i]`
fn matrix_trace(mat: &CompressedSparseData2<f32>) -> f32 {
    let n = mat.shape.0.min(mat.shape.1);
    let mut trace = 0.0;

    for i in 0..n {
        let row_start = mat.indptr[i];
        let row_end = mat.indptr[i + 1];

        for idx in row_start..row_end {
            if mat.indices[idx] == i {
                trace += mat.data[idx];
                break;
            }
        }
    }

    trace
}

/// Compute adaptive anisotropic diffusion kernel
///
/// Implementation from palantir package.  Uses knn/3-th nearest neighbor
/// distance as adaptive bandwidth. For edge (i,j) with distance d:
/// weight = exp(-d/σᵢ)
///
/// ### Params
///
/// * `knn_indices` - kNN indices for each cell
/// * `knn_distances` - kNN distances for each cell
/// * `knn` - Number of nearest neighbours used
/// * `squared_dist` - Are the distances squared (squared Euclidean for
///   example).
///
/// ### Returns
///
/// Symmetric kernel matrix
pub fn compute_diffusion_kernel(
    knn_indices: &[Vec<usize>],
    knn_distances: &[Vec<f32>],
    knn: usize,
    squared_dist: bool,
) -> CompressedSparseData2<f32> {
    let n = knn_indices.len();
    let adaptive_k = (knn / 3).max(1);

    let adaptive_std: Vec<f32> = knn_distances
        .iter()
        .map(|dists| {
            let mut sorted = dists.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[adaptive_k - 1]
        })
        .collect();

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for (i, neighbours) in knn_indices.iter().enumerate() {
        for (idx, &j) in neighbours.iter().enumerate() {
            // need to square root here, as I am not doing this during kNN generation
            let dist = if squared_dist {
                knn_distances[i][idx].sqrt()
            } else {
                knn_distances[i][idx]
            };
            let weight = (-dist / adaptive_std[i]).exp();
            rows.push(i);
            cols.push(j);
            vals.push(weight);
        }
    }

    let w = coo_to_csr(&rows, &cols, &vals, (n, n));

    // symmetrise: kernel = W + W^T
    let w_t = w.transpose_and_convert();

    sparse_add_csr(&w, &w_t)
}

/// Compute diffusion maps from kernel matrix
///
/// Normalises kernel to transition matrix and performs eigendecomposition.
///
/// ### Params
///
/// * `kernel` - Symmetric kernel matrix
/// * `n_components` - Number of eigenvectors to compute
///
/// ### Returns
///
/// (eigenvalues, eigenvectors) where eigenvectors is (n × n_components)
pub fn diffusion_map_from_kernel(
    kernel: &mut CompressedSparseData2<f32>,
    n_components: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<Vec<f32>>), BixverseErrors> {
    // Compute row sums (degrees)
    let row_sums: Vec<f32> = (0..kernel.shape.0)
        .map(|i| {
            (kernel.indptr[i]..kernel.indptr[i + 1])
                .map(|idx| kernel.data[idx])
                .sum()
        })
        .collect();

    // symmetric normalisation: D^(-1/2) * K * D^(-1/2)
    for i in 0..kernel.shape.0 {
        let d_i_sqrt = row_sums[i].sqrt();
        for idx in kernel.indptr[i]..kernel.indptr[i + 1] {
            let j = kernel.indices[idx];
            let d_j_sqrt = row_sums[j].sqrt();
            kernel.data[idx] /= d_i_sqrt * d_j_sqrt;
        }
    }

    compute_largest_eigenpairs_lanczos(kernel, n_components, seed)
}

/// Determine multiscale space by scaling eigenvectors
///
/// Scales eigenvectors by λᵢ/(1-λᵢ) for diffusion distance metric.
///
/// ### Params
///
/// * `eigenvalues` - Eigenvalues from diffusion maps
/// * `eigenvectors` - Eigenvectors (n × n_components)
/// * `n_eigs` - Optional number of eigenvectors to use (None = auto-detect via
///   eigengap)
///
/// ### Returns
///
/// Scaled eigenvectors (n × n_eigs)
pub fn determine_multiscale_space(
    eigenvalues: &[f32],
    eigenvectors: &[Vec<f32>],
    n_eigs: Option<usize>,
) -> Vec<Vec<f32>> {
    let n = eigenvectors.len();

    // auto-detect n_eigs using eigengap if not provided
    let use_n_eigs = if let Some(n) = n_eigs {
        n
    } else {
        let gaps: Vec<f32> = eigenvalues.windows(2).map(|w| w[0] - w[1]).collect();

        let max_gap_idx = gaps
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx + 1)
            .unwrap_or(3);

        max_gap_idx.max(3).min(eigenvalues.len())
    };

    let use_indices: Vec<usize> = (1..use_n_eigs).collect();

    let mut scaled = vec![vec![0.0f32; use_indices.len()]; n];

    for (out_idx, &eig_idx) in use_indices.iter().enumerate() {
        let lambda = eigenvalues[eig_idx];
        let scale = lambda / (1.0 - lambda);

        for i in 0..n {
            scaled[i][out_idx] = eigenvectors[i][eig_idx] * scale;
        }
    }

    scaled
}

/// Max-min waypoint sampling
///
/// For each dimension, iteratively selects points maximizing the minimum
/// distance to already selected points.
///
/// ### Params
///
/// * `data` - Multiscale space (n × n_dims)
/// * `num_waypoints` - Target number of waypoints
/// * `seed` - Random seed for initial point selection
///
/// ### Returns
///
/// Indices of selected waypoints
fn max_min_sampling(data: &[Vec<f32>], num_waypoints: usize, seed: u64) -> Vec<usize> {
    let n = data.len();
    let n_dims = data[0].len();
    let no_iterations = (num_waypoints / n_dims).max(1);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut waypoint_set = FxHashSet::default();

    for dim in 0..n_dims {
        let vec: Vec<f32> = data.iter().map(|row| row[dim]).collect();
        let mut iter_set = vec![rng.random_range(0..n)];
        let mut min_dists = vec![f32::MAX; n];

        // initialize distances to first point
        for i in 0..n {
            min_dists[i] = (vec[i] - vec[iter_set[0]]).abs();
        }

        // iteratively select maximally distant points
        for _ in 1..no_iterations {
            let new_wp = min_dists
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            iter_set.push(new_wp);

            for i in 0..n {
                let dist = (vec[i] - vec[new_wp]).abs();
                min_dists[i] = min_dists[i].min(dist);
            }
        }

        waypoint_set.extend(iter_set);
    }

    waypoint_set.into_iter().collect()
}

///////////////
// Landmarks //
///////////////

/// Density-weighted landmark selection (sample proportional to sqrt(degree))
///
/// ### Params
///
/// * `kernel` - Symmetric kernel matrix
/// * `n_landmarks` - Number of landmarks to extract
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// Indices of the landmarks
pub fn select_density_landmarks(
    kernel: &CompressedSparseData2<f32>,
    n_landmarks: usize,
    seed: u64,
) -> Vec<usize> {
    let n = kernel.shape.0;
    let weights: Vec<f64> = (0..n)
        .map(|i| {
            let s: f32 = (kernel.indptr[i]..kernel.indptr[i + 1])
                .map(|idx| kernel.data[idx])
                .sum();
            (s as f64).max(0.0).sqrt()
        })
        .collect();

    let mut rng = StdRng::seed_from_u64(seed);
    let dist = WeightedIndex::new(&weights).expect("invalid weights for density landmarks");
    let target = n_landmarks.min(n);
    let mut set = FxHashSet::default();
    while set.len() < target {
        set.insert(dist.sample(&mut rng));
    }
    set.into_iter().collect()
}

/// Pairwise kNN among landmarks in PCA space (or other embeddings)
///
/// ### Params
///
/// * `pca` - The PCA embedding (or any other provided embedding)
/// * `landmark_indices` - The landmark indices
/// * `k` - Number of neighbours to return
/// * `knn_params` - Reference to [KnnParams] for the (approximate) nearest
///   neighbour searches.
/// * `seed` - Seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The `(indices, distances)`
pub fn landmark_knn(
    pca: MatRef<f32>,
    landmark_indices: &[usize],
    k: usize,
    knn_params: &KnnParams,
    seed: usize,
    verbose: usize,
) -> ScKnnResults {
    let verbosity = parse_verbosity_level(verbose);

    let l = landmark_indices.len();
    let dim = pca.ncols();

    let landmark_mat = Mat::<f32>::from_fn(l, dim, |i, j| *pca.get(landmark_indices[i], j));

    let mut params = knn_params.clone();
    params.k = k.min(l.saturating_sub(1)).max(1);

    let (indices, distances) = generate_knn_with_dist(
        landmark_mat.as_ref(),
        &params,
        true,
        false,
        seed,
        verbosity.detailed_verbosity(),
    )?;

    Ok((indices, distances.expect("distances must be present")))
}

/// Row-stochastic N×L transitions in PCA space (Gaussian, adaptive bandwidth)
///
/// ### Params
///
/// * `pca` - The PCA embedding (or any other provided embedding)
/// * `landmark_indices` - The landmark indices
/// * `k` - Number of neighbours to return
/// * `bandwidth_scale` - The bandwidth scale
/// * `thresh` - The threshold
///
/// ### Returns
///
/// The landmark transition matrix
pub fn build_data_to_landmark_transitions(
    pca: MatRef<f32>,
    landmark_indices: &[usize],
    k: usize,
    bandwidth_scale: f32,
    thresh: f32,
) -> CompressedSparseData2<f32> {
    let n = pca.nrows();
    let l = landmark_indices.len();
    let dim = pca.ncols();
    let k_used = k.min(l).max(1);
    let bw_factor = bandwidth_scale * bandwidth_scale;

    // contiguous landmark coords for cache locality
    let landmark_coords: Vec<f32> = landmark_indices
        .iter()
        .flat_map(|&li| (0..dim).map(move |c| *pca.get(li, c)))
        .collect();

    let per_row: Vec<(Vec<usize>, Vec<f32>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut dists: Vec<(usize, f32)> = (0..l)
                .map(|li_idx| {
                    let mut d = 0.0f32;
                    for c in 0..dim {
                        let diff = pca.get(i, c) - landmark_coords[li_idx * dim + c];
                        d += diff * diff;
                    }
                    (li_idx, d)
                })
                .collect();

            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.truncate(k_used);

            let bw_raw = dists.last().map(|&(_, d)| d).unwrap_or(0.0);
            let bw = (bw_raw * bw_factor).max(f32::EPSILON);

            let mut kept: Vec<(usize, f32)> = dists
                .into_iter()
                .filter_map(|(li_idx, d)| {
                    let w = (-d / bw).exp();
                    (w >= thresh).then_some((li_idx, w))
                })
                .collect();

            let s: f32 = kept.iter().map(|&(_, w)| w).sum();
            if s > 0.0 {
                for (_, w) in &mut kept {
                    *w /= s;
                }
            }

            kept.sort_by_key(|&(li_idx, _)| li_idx);
            let idx: Vec<usize> = kept.iter().map(|&(li_idx, _)| li_idx).collect();
            let val: Vec<f32> = kept.iter().map(|&(_, w)| w).collect();
            (idx, val)
        })
        .collect();

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for (cell_id, (idx, val)) in per_row.into_iter().enumerate() {
        for (j, w) in idx.into_iter().zip(val) {
            rows.push(cell_id);
            cols.push(j);
            vals.push(w);
        }
    }
    coo_to_csr(&rows, &cols, &vals, (n, l))
}

/// Nystroem extension: y(i)[d] = (1/λ_d) · Σ_l P_nl[i,l] · y_landmark[l][d]
///
/// ### Params
///
/// * `p_nl` -
/// * `landmark_embedding` -
/// * `lambdas` -
///
/// ### Returns
fn nystrom_extend(
    p_nl: &CompressedSparseData2<f32>,
    landmark_embedding: &[Vec<f32>],
    lambdas: &[f32],
) -> Vec<Vec<f32>> {
    let n = p_nl.shape.0;
    let n_dim = landmark_embedding[0].len();

    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![0.0f32; n_dim];
            let start = p_nl.indptr[i];
            let end = p_nl.indptr[i + 1];
            for idx in start..end {
                let l = p_nl.indices[idx];
                let w = p_nl.data[idx];
                for d in 0..n_dim {
                    row[d] += w * landmark_embedding[l][d];
                }
            }
            for d in 0..n_dim {
                if lambdas[d].abs() > f32::EPSILON {
                    row[d] /= lambdas[d];
                }
            }
            row
        })
        .collect()
}

////////////////////
// Matrix updates //
////////////////////

/// Per-cell Frank-Wolfe argmins for the A update, one gradient column at a
/// time so the full k × n gradient is never materialised. The factor of 2 in
/// the true gradient is dropped: irrelevant to an argmin.
///
/// ### Params
///
/// * `t1` - k × k matrix Bᵀ K² B (symmetric)
/// * `a` - k × n current assignment matrix
/// * `k2_b` - n × k matrix K² B (rows are columns of t2)
/// * `k` - Number of archetypes
/// * `n` - Number of cells
///
/// ### Returns
///
/// Vector of length n with the argmin archetype index for each cell
fn fw_argmins_a(
    t1: &CompressedSparseData2<f32>,   // k × k, Bᵀ K² B (symmetric)
    a: &CompressedSparseData2<f32>,    // k × n, current A
    k2_b: &CompressedSparseData2<f32>, // n × k, K² B (rows are t2's columns)
    k: usize,
    n: usize,
) -> Vec<usize> {
    let a_t = a.transpose_and_convert(); // n × k: row j is column j of A

    const CHUNK: usize = 256;
    let n_chunks = n.div_ceil(CHUNK);

    let chunks: Vec<Vec<usize>> = (0..n_chunks)
        .into_par_iter()
        .map_init(
            || vec![0.0f32; k],
            |buf, chunk_idx| {
                let start = chunk_idx * CHUNK;
                let end = ((chunk_idx + 1) * CHUNK).min(n);
                let mut out = Vec::with_capacity(end - start);

                for j in start..end {
                    buf.fill(0.0);

                    for ai in a_t.indptr[j]..a_t.indptr[j + 1] {
                        let r = a_t.indices[ai];
                        let w = a_t.data[ai];
                        for ti in t1.indptr[r]..t1.indptr[r + 1] {
                            buf[t1.indices[ti]] += w * t1.data[ti];
                        }
                    }
                    for ki in k2_b.indptr[j]..k2_b.indptr[j + 1] {
                        buf[k2_b.indices[ki]] -= k2_b.data[ki];
                    }

                    let mut min_val = buf[0];
                    let mut min_idx = 0;
                    for i in 1..k {
                        if buf[i] < min_val {
                            min_val = buf[i];
                            min_idx = i;
                        }
                    }
                    out.push(min_idx);
                }
                out
            },
        )
        .collect();

    chunks.into_iter().flatten().collect()
}

/// Per-archetype Frank-Wolfe argmins and FW duality gap for the B update,
/// one gradient column at a time. Factor of 2 dropped: cancels in the gap
/// ratio, irrelevant to the argmin.
///
/// ### Params
///
/// * `k2_b_t` - k × n matrix, rows are columns of K² B
/// * `t1` - k × k matrix A Aᵀ (symmetric)
/// * `t2_t` - k × n matrix, rows are columns of K² Aᵀ
/// * `b_t` - k × n matrix, rows are columns of B
/// * `n` - Number of cells
/// * `k` - Number of archetypes
///
/// ### Returns
///
/// Tuple of argmin archetype index per archetype and the absolute FW duality
/// gap
fn fw_argmins_b(
    k2_b_t: &CompressedSparseData2<f32>, // k × n, rows are columns of K²B
    t1: &CompressedSparseData2<f32>,     // k × k, A Aᵀ (symmetric)
    t2_t: &CompressedSparseData2<f32>,   // k × n, rows are columns of K²Aᵀ
    b_t: &CompressedSparseData2<f32>,    // k × n, rows are columns of B
    n: usize,
    k: usize,
) -> (Vec<usize>, f32) {
    const CHUNK: usize = 64;
    let n_chunks = k.div_ceil(CHUNK);

    let chunks: Vec<(Vec<usize>, f32, f32)> = (0..n_chunks)
        .into_par_iter()
        .map_init(
            || vec![0.0f32; n],
            |buf, chunk_idx| {
                let start = chunk_idx * CHUNK;
                let end = ((chunk_idx + 1) * CHUNK).min(k);
                let mut amins = Vec::with_capacity(end - start);
                let mut g_dot_b = 0.0f32;
                let mut g_dot_e = 0.0f32;

                for c in start..end {
                    buf.fill(0.0);

                    // (K²B · t1)[:, c] = sum_m t1[m,c] * (K²B)[:, m]
                    for ti in t1.indptr[c]..t1.indptr[c + 1] {
                        let m = t1.indices[ti];
                        let tmc = t1.data[ti];
                        for ki in k2_b_t.indptr[m]..k2_b_t.indptr[m + 1] {
                            buf[k2_b_t.indices[ki]] += tmc * k2_b_t.data[ki];
                        }
                    }
                    for si in t2_t.indptr[c]..t2_t.indptr[c + 1] {
                        buf[t2_t.indices[si]] -= t2_t.data[si];
                    }

                    let mut min_val = buf[0];
                    let mut min_idx = 0;
                    for r in 1..n {
                        if buf[r] < min_val {
                            min_val = buf[r];
                            min_idx = r;
                        }
                    }
                    amins.push(min_idx);
                    g_dot_e += min_val;

                    for bi in b_t.indptr[c]..b_t.indptr[c + 1] {
                        g_dot_b += b_t.data[bi] * buf[b_t.indices[bi]];
                    }
                }
                (amins, g_dot_b, g_dot_e)
            },
        )
        .collect();

    let mut argmins = Vec::with_capacity(k);
    let mut g_dot_b = 0.0f32;
    let mut g_dot_e = 0.0f32;
    for (amins, gb, ge) in chunks {
        argmins.extend(amins);
        g_dot_b += gb;
        g_dot_e += ge;
    }

    (argmins, (g_dot_b - g_dot_e).abs())
}

//////////
// Main //
//////////

/// CPU implementation of the SEACells algorithm
///
/// SEACells identifies metacells (groupings of similar cells) using kernel
/// archetypal analysis. The algorithm solves a convex optimisation problem to
/// find archetypes that minimise reconstruction error whilst maintaining
/// sparsity.
///
/// This Rust implementation includes memory optimisations:
///
/// - Never materialises K_square, instead computing K @ (K @ X) on the fly
/// - Prunes small values to maintain sparsity
/// - Supports fast random initialisation for large datasets
pub struct SEACells<'a> {
    /// Number of cells in the dataset.
    n_cells: usize,
    /// Sparse symmetric kernel matrix K.
    kernel_mat: Option<CompressedSparseData2<f32>>,
    /// Assignment matrix (k × n) mapping cells to SEACells.
    a: Option<CompressedSparseData2<f32>>,
    /// Archetype matrix (n × k) defining SEACells as cell combinations.
    b: Option<CompressedSparseData2<f32>>,
    /// Indices of cells selected as initial archetypes.
    archetypes: Option<Vec<usize>>,
    /// Residual sum of squares at each iteration.
    rss_history: Vec<f32>,
    /// Absolute RSS change threshold for convergence.
    convergence_threshold: Option<f32>,
    ///  Cached ||K||_F^2 for trace-based RSS.
    k_frobenius_norm_sq: Option<f32>,
    /// SEACell parameters.
    params: &'a SEACellsParams,
}

impl<'a> SEACells<'a> {
    /// Create a new SEACells instance
    ///
    /// ### Params
    ///
    /// * `n_cells` - Number of cells in the dataset
    /// * `params` - Algorithm parameters
    ///
    /// ### Returns
    ///
    /// New `SEACells` instance with uninitialised matrices
    pub fn new(n_cells: usize, params: &'a SEACellsParams) -> Self {
        Self {
            n_cells,
            kernel_mat: None,
            a: None,
            b: None,
            archetypes: None,
            convergence_threshold: None,
            k_frobenius_norm_sq: None,
            rss_history: Vec::new(),
            params,
        }
    }

    /// Construct the kernel matrix from k-NN graph with adaptive RBF weights
    ///
    /// Builds a sparse symmetric kernel K where `K[i,j] ∝ similarity` between
    /// cells i and j:
    ///
    /// ```K[i,j] = exp(-||xᵢ - xⱼ||² / (σᵢ σⱼ))```
    ///
    /// where σᵢ is the median k-NN distance for cell i (taken from the
    /// already-squared `knn_distances` via .sqrt() at the median index).
    ///
    /// The graph is first symmetrised by union (edge if either direction
    /// exists) or intersection (edge only if both directions exist); self-loops
    /// are added before weights are computed, which guarantees `K[i,i] = 1` and
    /// a symmetric sparsity pattern. Weights are symmetric by construction
    /// since both the distance and σᵢσⱼ are symmetric.
    ///
    /// K² is never materialised - downstream operations compute
    /// K @ (K @ X), bounding memory to O(nnz(K)).
    ///
    /// ### Params
    ///
    /// * `pca` - PCA/SVD matrix (n_cells × n_components)
    /// * `knn_indices` - k-NN indices for each cell
    /// * `knn_distances` - k-NN distances for each cell
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    pub fn construct_kernel_mat(
        &mut self,
        pca: MatRef<f32>,
        knn_indices: &[Vec<usize>],
        knn_distances: &[Vec<f32>],
        verbose: usize,
    ) {
        let verbosity = parse_verbosity_level(verbose);

        let n = pca.nrows();
        let k = knn_indices[0].len();

        if verbosity.normal_verbosity() {
            println!("Computing adaptive bandwidth RBF kernel...");
        }

        let graph_construction =
            parse_knn_symmetrisation(&self.params.graph_building).unwrap_or_default();

        let median_idx = k / 2;
        let median_dist = knn_distances
            .iter()
            .map(|d| d[median_idx].sqrt())
            .collect::<Vec<f32>>();

        let mut edges = FxHashSet::default();
        for (i, neighbours) in knn_indices.iter().enumerate() {
            for &j in neighbours {
                edges.insert((i, j));
            }
        }

        match graph_construction {
            KnnSymmetrisation::Union => {
                let to_add: Vec<_> = edges
                    .iter()
                    .filter_map(|&(i, j)| (!edges.contains(&(j, i))).then_some((j, i)))
                    .collect();
                edges.extend(to_add);
            }
            KnnSymmetrisation::Intersection => {
                let to_keep: FxHashSet<_> = edges
                    .iter()
                    .copied()
                    .filter(|&(i, j)| edges.contains(&(j, i)))
                    .collect();
                edges = to_keep;
            }
        }

        for i in 0..n {
            edges.insert((i, i));
        }

        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f32> = Vec::new();

        for &(i, j) in &edges {
            let mut dist_square = 0_f32;
            for dim in 0..pca.ncols() {
                let diff = pca.get(i, dim) - pca.get(j, dim);
                dist_square += diff * diff;
            }
            let sigma_prod = median_dist[i] * median_dist[j];
            let val = (-dist_square / sigma_prod).exp();

            rows.push(i);
            cols.push(j);
            vals.push(val);
        }

        if verbosity.normal_verbosity() {
            println!(
                "Built kernel with {} non-zeros",
                vals.len().separate_with_underscores()
            );
        }

        let kernel = coo_to_csr(&rows, &cols, &vals, (n, n));

        if self.n_cells > 20000 {
            if verbosity.detailed_verbosity() {
                println!(" Pre-computing kernel Frobenius norm...");
            }
            let k_frob = frobenius_norm(&kernel);
            self.k_frobenius_norm_sq = Some(k_frob * k_frob);
        }

        self.kernel_mat = Some(kernel);
    }

    /// Compute K² @ X = K @ (K @ X) for a sparse matrix X
    ///
    /// Avoids materialising K² entirely. The intermediate result K @ X has
    /// the same shape as X and remains sparse when X is sparse, keeping
    /// memory bounded to O(nnz(K)).
    ///
    /// K² arises naturally in the FW gradients because the objective
    /// `||K - KBA||_F²` has Kᵀ K = K² in its normal equations (K is
    /// symmetric here).
    ///
    /// ### Params
    ///
    /// * `x` - Sparse matrix to multiply
    ///
    /// ### Returns
    ///
    /// Result of K^2 @ X
    fn k_squared_matmul(
        &self,
        x: &CompressedSparseData2<f32>,
    ) -> Result<CompressedSparseData2<f32>, BixverseErrors> {
        if self.kernel_mat.is_none() {
            return Err(BixverseErrors::SEACellsKernelMatrixMissing);
        }

        let k = self.kernel_mat.as_ref().unwrap();
        let kx = csr_matmul_csr(k, x);
        Ok(csr_matmul_csr(k, &kx))
    }

    /// Compute K^2 @ v = K @ (K @ v) for a dense vector v
    ///
    /// ### Params
    ///
    /// * `v` - Dense vector to multiply
    ///
    /// ### Returns
    ///
    /// Result of K^2 @ v as a dense vector
    fn k_squared_matvec(&self, v: &[f32]) -> Vec<f32> {
        let k = self.kernel_mat.as_ref().unwrap();
        let kv = csr_matvec(k, v);
        csr_matvec(k, &kv)
    }

    /// Fit the SEACells model
    ///
    /// Runs the main optimisation loop:
    ///
    /// 1. Initialises archetypes (greedy CSSP or random)
    /// 2. Initialises A and B matrices
    /// 3. Alternates updating A and B using Frank-Wolfe until convergence
    ///
    /// Convergence is reached when RSS change < epsilon × RSS(0), subject to
    /// minimum iteration requirements.
    ///
    /// ### Params
    ///
    /// * `seed` - Random seed for reproducibility
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    pub fn fit(&mut self, seed: usize, verbose: usize) -> Result<(), BixverseErrors> {
        let verbosity = parse_verbosity_level(verbose);

        if self.kernel_mat.is_none() {
            return Err(BixverseErrors::SEACellsKernelMatrixMissing);
        }
        if self.archetypes.is_none() {
            return Err(BixverseErrors::SEACellsArchetypesMissing);
        }

        self.initialise_matrices(verbose, seed as u64)?;

        let a = self.a.as_ref().unwrap();
        let b = self.b.as_ref().unwrap();

        let initial_rss = self.compute_rss(a, b)?;
        self.rss_history.push(initial_rss);
        self.convergence_threshold = Some(self.params.convergence_epsilon * initial_rss);

        if verbosity.normal_verbosity() {
            println!("Initial RSS: {:.6}", initial_rss);
            println!(
                "Convergence threshold: {:.6}",
                self.convergence_threshold.unwrap()
            );
        }

        let mut converged = false;
        let mut n_iter = 0;

        while (!converged && n_iter < self.params.max_iter) || n_iter < self.params.min_iter {
            let iter_start = Instant::now();
            n_iter += 1;

            let b_current = self.b.take().unwrap();
            let a_current = self.a.take().unwrap();

            let a_new = self.update_a_mat(&b_current, &a_current, verbose)?;
            let b_new = self.update_b_mat(&a_new, &b_current, verbose)?;

            let rss = self.compute_rss(&a_new, &b_new)?;
            self.rss_history.push(rss);

            self.a = Some(a_new);
            self.b = Some(b_new);

            let iter_duration = iter_start.elapsed();

            if verbosity.normal_verbosity() {
                println!(
                    "Iteration {}: RSS = {:.6}, Time = {:.2}s",
                    n_iter,
                    rss,
                    iter_duration.as_secs_f32()
                );
            }

            if n_iter > 1 {
                let rss_diff = (self.rss_history[n_iter - 1] - self.rss_history[n_iter]).abs();
                if rss_diff < self.convergence_threshold.unwrap() && n_iter >= self.params.min_iter
                {
                    if verbosity.normal_verbosity() {
                        println!("Converged after {} iterations!", n_iter);
                    }
                    converged = true;
                }
            }
        }

        if !converged && verbosity.normal_verbosity() {
            println!(
                "Warning: Algorithm did not converge after {} iterations",
                self.params.max_iter
            );
        }

        Ok(())
    }

    /// Initialise archetypes using adaptive strategy
    ///
    /// For small datasets (< greedy_threshold): combines waypoint + greedy CSSP
    /// For large datasets (>= greedy_threshold): uses fast random initialisation
    ///
    /// ### Params
    ///
    /// * `knn_indices` - k-NN indices for each cell
    /// * `knn_distances` - k-NN distances for each cell
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    /// * `squared_dist` - Are the distances squared (squared Euclidean for
    ///   example).
    /// * `seed` - Random seed for initialisation
    pub fn initialise_archetypes(
        &mut self,
        knn_indices: &[Vec<usize>],
        knn_distances: &[Vec<f32>],
        verbose: usize,
        squared_dist: bool,
        seed: u64,
    ) -> Result<(), BixverseErrors> {
        let verbosity = parse_verbosity_level(verbose);

        if self.n_cells > self.params.greedy_threshold {
            if verbosity.normal_verbosity() {
                println!(
                    "Dataset large (n={}), using fast random init (threshold: {})",
                    self.n_cells.separate_with_underscores(),
                    self.params.greedy_threshold
                );
            }
            self.initialise_archetypes_random(verbose, seed);
        } else {
            self.initialise_archetypes_combined(
                knn_indices,
                knn_distances,
                squared_dist,
                verbose,
                seed,
            )?;
        }
        Ok(())
    }

    /// Fast random archetype initialisation
    ///
    /// Randomly samples k cells as initial archetypes. Used for large datasets
    /// where greedy CSSP is computationally expensive.
    ///
    /// ### Params
    ///
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    /// * `seed` - Random seed for reproducibility
    fn initialise_archetypes_random(&mut self, verbose: usize, seed: u64) {
        let verbosity = parse_verbosity_level(verbose);

        let mut rng = StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..self.n_cells).collect();
        indices.shuffle(&mut rng);

        let archetypes: Vec<usize> = indices.into_iter().take(self.params.n_sea_cells).collect();

        if verbosity.normal_verbosity() {
            println!("Selected {} random archetypes", archetypes.len());
        }

        self.archetypes = Some(archetypes);
    }

    /// Combined waypoint + greedy initialisation (matches Python logic)
    ///
    /// 1. Gets waypoint centres (may return < k cells)
    /// 2. Tops up with greedy CSSP to reach k cells
    /// 3. Deduplicates and takes first k unique cells
    ///
    /// ### Params
    ///
    /// * `knn_indices` - k-NN indices for each cell
    /// * `knn_distances` - k-NN distances for each cell
    /// * `squared_dist` - Are the distances squared (squared Euclidean for
    ///   example).
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    /// * `seed` - Random seed for waypoint sampling
    fn initialise_archetypes_combined(
        &mut self,
        knn_indices: &[Vec<usize>],
        knn_distances: &[Vec<f32>],
        squared_dist: bool,
        verbose: usize,
        seed: u64,
    ) -> Result<(), BixverseErrors> {
        let verbosity = parse_verbosity_level(verbose);
        let k = self.params.n_sea_cells;

        if verbosity.normal_verbosity() {
            println!("Computing diffusion maps for waypoint initialisation...");
        }

        let mut kernel = compute_diffusion_kernel(
            knn_indices,
            knn_distances,
            self.params.knn_params.k,
            squared_dist,
        );

        let (eigenvalues, eigenvectors) =
            diffusion_map_from_kernel(&mut kernel, self.params.knn_params.k, seed)?;

        let multiscale = determine_multiscale_space(&eigenvalues, &eigenvectors, Some(10));
        let waypoint_ix = max_min_sampling(&multiscale, k, seed);

        if verbosity.normal_verbosity() {
            println!(
                "Selecting {} cells from waypoint initialisation.",
                waypoint_ix.len()
            );
        }

        let from_greedy = k.saturating_sub(waypoint_ix.len());

        if verbosity.normal_verbosity() {
            println!("Initialising residual matrix using greedy column selection");
        }
        let greedy_ix = self.get_greedy_centres(from_greedy + 10);

        if verbosity.normal_verbosity() {
            println!(
                "Selecting {} cells from greedy initialisation.",
                from_greedy
            );
        }

        let mut all_ix = waypoint_ix;
        all_ix.extend(greedy_ix);

        let mut seen = FxHashSet::default();
        let unique_ix: Vec<usize> = all_ix
            .into_iter()
            .filter(|&x| seen.insert(x))
            .take(k)
            .collect();

        self.archetypes = Some(unique_ix);
        Ok(())
    }

    /// Landmark-based archetype initialisation for large datasets
    ///
    /// Density-samples L landmarks, builds a small L×L diffusion operator,
    /// eigendecomposes that, then Nystroem-extends the multiscale embedding
    /// to all N cells before max-min waypoint sampling.
    ///
    /// Skips greedy CSSP top-up because its initialisation phase is O(N²);
    /// pads with random cells if waypoint dedup falls short.
    ///
    /// ### Params
    ///
    /// * `pca` - PCA/SVD matrix (n_cells × n_components)
    /// * `knn_indices` - kNN indices for each cell
    /// * `knn_distances` - kNN distances for each cell
    /// * `squared_dist` - Are the distances squared (squared Euclidean for
    ///   example).
    /// * `n_landmarks` - Number of landmarks (typically 5-10× n_sea_cells)
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    /// * `seed` - Random seed for reproducibility
    #[allow(clippy::too_many_arguments)]
    pub fn initialise_archetypes_landmark(
        &mut self,
        pca: MatRef<f32>,
        knn_indices: &[Vec<usize>],
        knn_distances: &[Vec<f32>],
        squared_dist: bool,
        n_landmarks: usize,
        verbose: usize,
        seed: u64,
    ) -> Result<(), BixverseErrors> {
        let verbosity = parse_verbosity_level(verbose);

        let k = self.params.n_sea_cells;
        let n = self.n_cells;
        let knn_k = self.params.knn_params.k;

        if verbosity.normal_verbosity() {
            println!("Building diffusion kernel for landmark selection...");
        }
        let kernel = compute_diffusion_kernel(knn_indices, knn_distances, knn_k, squared_dist);

        if verbosity.normal_verbosity() {
            println!(
                "Selecting {} density-weighted landmarks...",
                n_landmarks.separate_with_underscores()
            );
        }
        let landmark_indices = select_density_landmarks(&kernel, n_landmarks, seed);
        let l = landmark_indices.len();

        let k_ll = knn_k.min(l.saturating_sub(1)).max(3);
        if verbosity.normal_verbosity() {
            println!("Building landmark-landmark diffusion operator (L={})...", l);
        }
        let (ll_idx, ll_dist) = landmark_knn(
            pca,
            &landmark_indices,
            k_ll,
            &self.params.knn_params,
            seed as usize,
            verbose,
        )?;
        let squared_dist = self.params.knn_params.ann_dist == "euclidean";

        let mut ll_kernel = compute_diffusion_kernel(&ll_idx, &ll_dist, k_ll, squared_dist);

        let n_eigs = k_ll.min(l - 1).max(11);
        let (evals, evecs) = diffusion_map_from_kernel(&mut ll_kernel, n_eigs, seed)?;

        let landmark_multiscale = determine_multiscale_space(&evals, &evecs, Some(10));
        let n_components = landmark_multiscale[0].len();
        let used_lambdas: Vec<f32> = (1..=n_components).map(|i| evals[i]).collect();

        if verbosity.normal_verbosity() {
            println!(
                "Building data-to-landmark transitions ({} × {})...",
                n.separate_with_underscores(),
                l
            );
        }
        let p_nl = build_data_to_landmark_transitions(pca, &landmark_indices, knn_k, 1.0, 1e-4);

        if verbosity.detailed_verbosity() {
            println!("Nystroem-extending multiscale embedding to full data...");
        }
        let multiscale = nystrom_extend(&p_nl, &landmark_multiscale, &used_lambdas);

        let waypoint_ix = max_min_sampling(&multiscale, k, seed);
        if verbosity.detailed_verbosity() {
            println!("Selected {} cells from waypoint init", waypoint_ix.len());
        }

        let mut seen = FxHashSet::default();
        let mut unique_ix: Vec<usize> = waypoint_ix
            .into_iter()
            .filter(|&x| seen.insert(x))
            .take(k)
            .collect();

        // Pad with random cells if waypoint dedup fell short
        if unique_ix.len() < k {
            if verbosity.normal_verbosity() {
                println!(
                    "Padding {} cells with random selection",
                    k - unique_ix.len()
                );
            }
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));
            let mut remaining: Vec<usize> = (0..n).filter(|i| !seen.contains(i)).collect();
            remaining.shuffle(&mut rng);
            while unique_ix.len() < k {
                match remaining.pop() {
                    Some(idx) => unique_ix.push(idx),
                    None => break,
                }
            }
        }

        self.archetypes = Some(unique_ix);

        Ok(())
    }

    /// Get greedy centres via chunked K^2 column computation
    ///
    /// Processes cells in chunks to bound peak memory to O(CHUNK_SIZE × n)
    /// rather than O(n^2). Each chunk computes K^2 @ e_i = K @ (K @ e_i) via
    /// two sparse matvecs, exploiting K's symmetry to extract column i from
    /// its CSR rows directly.
    ///
    /// ### Params
    ///
    /// * `n_centres` - Number of centres to select
    ///
    /// ### Returns
    ///
    /// Vector of selected cell indices
    fn get_greedy_centres(&self, n_centres: usize) -> Vec<usize> {
        let kernel = self.kernel_mat.as_ref().unwrap();
        let n = kernel.shape.0;

        const INIT_CHUNK_SIZE: usize = 256;
        const TILE: usize = 4096;

        let mut f = vec![0_f32; n];
        let mut g = vec![0_f32; n];

        // Initial f[i] = sum_j (K^2[j,i])^2, g[i] = K^2[i,i]
        for chunk_start in (0..n).step_by(INIT_CHUNK_SIZE) {
            let chunk_end = (chunk_start + INIT_CHUNK_SIZE).min(n);

            let chunk_results: Vec<(usize, Vec<f32>)> = (chunk_start..chunk_end)
                .into_par_iter()
                .map(|i| {
                    let mut row_i = vec![0_f32; n];
                    for idx in kernel.indptr[i]..kernel.indptr[i + 1] {
                        row_i[kernel.indices[idx]] = kernel.data[idx];
                    }
                    let k2_col_i = csr_matvec(kernel, &row_i);
                    (i, k2_col_i)
                })
                .collect();

            for (i, k2_col_i) in chunk_results {
                g[i] = k2_col_i[i];
                for j in 0..n {
                    f[j] += k2_col_i[j] * k2_col_i[j];
                }
            }
        }

        let mut omega: Vec<Vec<f32>> = vec![vec![0_f32; n]; n_centres];
        let mut centres: Vec<usize> = Vec::with_capacity(n_centres);

        let mut e_p = vec![0.0f32; n];
        let mut omega_new = vec![0.0f32; n];
        let mut pl = vec![0.0f32; n];

        for iter in 0..n_centres {
            // Argmax of f / g
            let best_idx = (0..n)
                .into_par_iter()
                .filter_map(|i| (g[i] > 1e-15).then(|| (i, f[i] / g[i])))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            centres.push(best_idx);

            // delta = K^2[:, p] - sum_{r<iter} omega[r][p] * omega[r]
            e_p.fill(0.0);
            e_p[best_idx] = 1.0;
            let mut delta = self.k_squared_matvec(&e_p);

            let delta_coefs: Vec<f32> = (0..iter).map(|r| omega[r][best_idx]).collect();
            delta
                .par_chunks_mut(TILE)
                .enumerate()
                .for_each(|(tile_idx, tile)| {
                    let start = tile_idx * TILE;
                    let end = start + tile.len();
                    for r in 0..iter {
                        let coef = delta_coefs[r];
                        if coef == 0.0 {
                            continue;
                        }
                        let omega_r = &omega[r][start..end];
                        for (d, o) in tile.iter_mut().zip(omega_r.iter()) {
                            *d -= coef * o;
                        }
                    }
                });

            delta[best_idx] = delta[best_idx].max(0.0);
            let delta_p_sqrt = delta[best_idx].sqrt().max(1e-6);

            omega_new
                .par_iter_mut()
                .zip(delta.par_iter())
                .for_each(|(o, &d)| *o = d / delta_p_sqrt);

            let omega_sq_norm: f32 = omega_new.par_iter().map(|&x| x * x).sum();
            let k_omega_new = self.k_squared_matvec(&omega_new);

            // pl[i] = sum_r <omega_r, omega_new> * omega_r[i]
            let omega_dot_new: Vec<f32> = (0..iter)
                .into_par_iter()
                .map(|r| {
                    omega[r]
                        .iter()
                        .zip(omega_new.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                })
                .collect();

            pl.fill(0.0);
            pl.par_chunks_mut(TILE)
                .enumerate()
                .for_each(|(tile_idx, tile)| {
                    let start = tile_idx * TILE;
                    let end = start + tile.len();
                    for r in 0..iter {
                        let coef = omega_dot_new[r];
                        if coef == 0.0 {
                            continue;
                        }
                        let omega_r = &omega[r][start..end];
                        for (p, o) in tile.iter_mut().zip(omega_r.iter()) {
                            *p += coef * o;
                        }
                    }
                });

            // Update f and g
            f.par_iter_mut()
                .zip(g.par_iter_mut())
                .zip(omega_new.par_iter())
                .zip(k_omega_new.par_iter())
                .zip(pl.par_iter())
                .for_each(|((((f_i, g_i), &o_i), &k_i), &p_i)| {
                    let omega_hadamard = o_i * o_i;
                    let term1 = omega_sq_norm * omega_hadamard;
                    let term2 = o_i * (k_i - p_i);
                    *f_i += -2.0 * term2 + term1;
                    *g_i += omega_hadamard;
                });

            omega[iter].copy_from_slice(&omega_new);
        }

        centres
    }

    /// Initialise A and B matrices
    ///
    /// Creates:
    ///
    /// - B matrix (n × k): one-hot encoding of archetype cells
    /// - A matrix (k × n): sparse random assignments, column-L1-normalised
    ///
    /// Each cell is randomly assigned to ⌈0.25 k⌉ archetypes with uniform
    /// random weights, then column-normalised so each cell's weights sum to 1.
    /// A is then refined by one full Frank-Wolfe update pass against the fixed
    /// B for a better starting point.
    ///
    /// Matches the Python reference, which uses the same 25%-of-k sparsity
    /// and L1 column normalisation.
    ///
    /// ### Params
    ///
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    /// * `seed` - Random seed for A matrix initialisation
    fn initialise_matrices(&mut self, verbose: usize, seed: u64) -> Result<(), BixverseErrors> {
        let verbosity = parse_verbosity_level(verbose);

        let archetypes = self.archetypes.as_ref().unwrap();
        let k = archetypes.len();
        let n = self.n_cells;

        if verbosity.normal_verbosity() {
            println!("Initialising A and B matrices...");
        }

        let mut b_rows = Vec::new();
        let mut b_cols = Vec::new();
        let mut b_vals = Vec::new();

        for (col, &row) in archetypes.iter().enumerate() {
            b_rows.push(row);
            b_cols.push(col);
            b_vals.push(1_f32);
        }

        let b = coo_to_csr(&b_rows, &b_cols, &b_vals, (n, k));

        // changed compared to the original Python
        let archetypes_per_cell = 10.min(k);
        let mut rng = StdRng::seed_from_u64(seed);

        let mut a_rows = Vec::new();
        let mut a_cols = Vec::new();
        let mut a_vals = Vec::new();

        for cell in 0..n {
            for _ in 0..archetypes_per_cell {
                let archetype = rng.random_range(0..k);
                a_rows.push(archetype);
                a_cols.push(cell);
                a_vals.push(rng.random::<f32>());
            }
        }

        let mut a = coo_to_csr(&a_rows, &a_cols, &a_vals, (k, n));
        normalise_csr_columns_l1(&mut a);

        a = self.update_a_mat(&b, &a, verbose)?;

        self.a = Some(a);
        self.b = Some(b);

        Ok(())
    }

    /// Update assignment matrix A using Frank-Wolfe algorithm
    ///
    /// Solves:
    ///
    /// ```min_A ||K - K B A||_F²```
    ///
    /// subject to A columns summing to 1 (column-stochastic).
    ///
    /// The gradient with respect to A is:
    ///
    /// ```∇_A = 2 (Bᵀ K² B A - Bᵀ K²) = 2 (t1 A - t2)```
    ///
    /// where:
    /// - t1 = Bᵀ K² B   [k × k]
    /// - t2 = Bᵀ K²     [k × n]
    ///
    /// K² @ B is computed as K @ (K @ B) without ever materialising K².
    ///
    /// For each cell (column), sets weight to 1 for the archetype with
    /// minimum gradient, then takes a convex step A ← (1 - γ) A + γ E
    /// with γ = 2/(t + 2).
    ///
    /// ### Params
    ///
    /// * `b` - Current archetype matrix
    /// * `a_prev` - Previous assignment matrix
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    ///
    /// ### Returns
    ///
    /// Updated assignment matrix
    fn update_a_mat(
        &self,
        b: &CompressedSparseData2<f32>,
        a_prev: &CompressedSparseData2<f32>,
        verbose: usize,
    ) -> Result<CompressedSparseData2<f32>, BixverseErrors> {
        let verbosity = parse_verbosity_level(verbose);

        let k2_b = self.k_squared_matmul(b)?;
        let t2 = k2_b.transpose_and_convert();
        let t1 = csr_matmul_csr(&t2, b);
        drop(t2);

        let mut a = a_prev.clone();
        let n = a.shape.1;
        let k = a.shape.0;

        for t in 0..self.params.max_fw_iters {
            let argmins = fw_argmins_a(&t1, &a, &k2_b, k, n);

            let mut e_data: Vec<(usize, usize, f32)> = argmins
                .iter()
                .enumerate()
                .map(|(col, &row)| (row, col, 1.0f32))
                .collect();
            e_data.sort_unstable_by_key(|&(r, c, _)| (r, c));

            let e_rows: Vec<usize> = e_data.iter().map(|&(r, _, _)| r).collect();
            let e_cols: Vec<usize> = e_data.iter().map(|&(_, c, _)| c).collect();
            let e_vals: Vec<f32> = e_data.iter().map(|&(_, _, v)| v).collect();
            let e = coo_to_csr_presorted(&e_rows, &e_cols, &e_vals, (k, n));

            let step_size = 2.0 / (t as f32 + 2.0);
            let retain = 1.0 - step_size;
            for val in &mut a.data {
                *val *= retain;
            }
            let e_scaled = sparse_scalar_multiply_csr(&e, step_size);
            a = sparse_add_csr(&a, &e_scaled);

            if self.params.pruning {
                prune_and_renormalise(&mut a, self.params.pruning_threshold);
            }

            if verbosity.detailed_verbosity() && (t + 1) % 10 == 0 {
                println!(
                    "  A matrix Frank-Wolfe iteration: {} / {}",
                    t + 1,
                    self.params.max_fw_iters
                );
            }
        }

        Ok(a)
    }

    /// Update archetype matrix B using Frank-Wolfe algorithm
    ///
    /// Solves:
    ///
    /// ```min_B ||K - K B A||_F²```
    ///
    /// subject to B columns summing to 1 (column-stochastic).
    ///
    /// The gradient with respect to B is:
    ///
    /// ```∇_B = 2 (K² B A Aᵀ - K² Aᵀ) = 2 (K² B · t1 - t2)```
    ///
    /// where:
    /// - t1 = A Aᵀ     [k × k]
    /// - t2 = K² Aᵀ    [n × k]
    ///
    /// K² @ B is recomputed each inner iteration as K @ (K @ B) because B
    /// is what is being updated. Two matmuls through sparse K is still
    /// cheaper than one through a materialised K² at single-cell scale.
    ///
    /// Includes early stopping when the Frank-Wolfe step contribution
    /// falls below FW_TOLERANCE after a minimum of 10 iterations.
    ///
    /// ### Params
    ///
    /// * `a` - Current assignment matrix
    /// * `b_prev` - Previous archetype matrix
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    ///
    /// ### Returns
    ///
    /// Updated archetype matrix
    fn update_b_mat(
        &self,
        a: &CompressedSparseData2<f32>,
        b_prev: &CompressedSparseData2<f32>,
        verbose: usize,
    ) -> Result<CompressedSparseData2<f32>, BixverseErrors> {
        let verbosity = parse_verbosity_level(verbose);

        const FW_REL_TOL: f32 = 1e-3;
        const MIN_FW_ITERS: usize = 10;

        let a_t = a.transpose_and_convert();
        let t1 = csr_matmul_csr(a, &a_t);
        let t2 = self.k_squared_matmul(&a_t)?;
        let t2_t = t2.transpose_and_convert();
        drop(t2);

        let mut b = b_prev.clone();
        let n = b.shape.0;
        let k = b.shape.1;
        let mut initial_gap: f32 = 0.0;

        for t in 0..self.params.max_fw_iters {
            let k2_b = self.k_squared_matmul(&b)?;
            let k2_b_t = k2_b.transpose_and_convert();
            drop(k2_b);
            let b_t = b.transpose_and_convert();

            let (argmins, fw_gap) = fw_argmins_b(&k2_b_t, &t1, &t2_t, &b_t, n, k);
            if t == 0 {
                initial_gap = fw_gap.max(1e-12);
            }

            let mut e_data: Vec<(usize, usize, f32)> = argmins
                .iter()
                .enumerate()
                .map(|(col, &row)| (row, col, 1.0f32))
                .collect();
            e_data.sort_unstable_by_key(|&(r, c, _)| (r, c));

            let e_rows: Vec<usize> = e_data.iter().map(|&(r, _, _)| r).collect();
            let e_cols: Vec<usize> = e_data.iter().map(|&(_, c, _)| c).collect();
            let e_vals: Vec<f32> = e_data.iter().map(|&(_, _, v)| v).collect();
            let e = coo_to_csr_presorted(&e_rows, &e_cols, &e_vals, (n, k));

            let step_size = 2.0 / (t as f32 + 2.0);
            let retain = 1.0 - step_size;
            for val in &mut b.data {
                *val *= retain;
            }
            let e_scaled = sparse_scalar_multiply_csr(&e, step_size);
            b = sparse_add_csr(&b, &e_scaled);

            if self.params.pruning {
                prune_and_renormalise(&mut b, self.params.pruning_threshold);
            }

            if verbosity.detailed_verbosity() && (t + 1) % 10 == 0 {
                println!(
                    "  B matrix Frank-Wolfe iteration: {} / {}",
                    t + 1,
                    self.params.max_fw_iters
                );
            }

            if fw_gap / initial_gap < FW_REL_TOL && t >= MIN_FW_ITERS {
                if verbosity.detailed_verbosity() {
                    println!(
                        "  B matrix FW converged at iter {} (gap: {:.4e})",
                        t + 1,
                        fw_gap
                    );
                }
                break;
            }
        }

        Ok(b)
    }

    /// Compute residual sum of squares (RSS)
    ///
    /// Returns the Frobenius norm (not squared) of the reconstruction
    /// residual:
    ///
    /// ```||K - K B A||_F```
    ///
    /// This matches the reference Python implementation
    /// (`np.linalg.norm` / `scipy.sparse.linalg.norm`, both of which
    /// default to the Frobenius norm, unsquared). The convergence check
    /// `|RSS_{i-1} - RSS_i| < ε · RSS_0` is therefore in norm units, not
    /// squared-norm units.
    ///
    /// ### Params
    ///
    /// * `a` - Assignment matrix
    /// * `b` - Archetype matrix
    ///
    /// ### Returns
    ///
    /// RSS value (lower is better fit)
    fn compute_rss(
        &self,
        a: &CompressedSparseData2<f32>,
        b: &CompressedSparseData2<f32>,
    ) -> Result<f32, BixverseErrors> {
        if self.n_cells <= 20000 {
            Ok(self.compute_rss_simple(a, b))
        } else {
            Ok(self.compute_rss_trace(a, b)?)
        }
    }

    /// Fast RSS computation for small datasets (materialises reconstruction)
    ///
    /// Directly forms the n × n reconstruction K B A and returns the Frobenius
    /// norm of (K - K B A). Cheap when n is small.
    ///
    /// ### Params
    ///
    /// * `a` - The A matrix
    /// * `b` - The B matrix
    ///
    /// ### Returns
    ///
    /// The residual sum of squares (RSS)
    fn compute_rss_simple(
        &self,
        a: &CompressedSparseData2<f32>,
        b: &CompressedSparseData2<f32>,
    ) -> f32 {
        let k_mat = self.kernel_mat.as_ref().unwrap();
        let k_b = csr_matmul_csr(k_mat, b);
        let reconstruction = csr_matmul_csr(&k_b, a);
        let diff = sparse_subtract_csr(k_mat, &reconstruction);
        frobenius_norm(&diff)
    }

    /// Memory-efficient RSS computation for large datasets (uses trace trick)
    ///
    /// Expands the squared Frobenius norm via the trace identity:
    ///
    /// ```||K - K B A||_F² = ||K||_F² - 2 tr(K² B A) + tr(A Aᵀ Bᵀ K² B)```
    ///
    /// Cyclic trace reordering keeps every intermediate at worst (n × k) or
    /// (k × k); the n × n reconstruction is never formed. All K² @ X terms are
    /// computed as K @ (K @ X).
    ///
    /// The final `.sqrt()` converts back to the Frobenius norm to match
    /// `compute_rss_simple`.
    ///
    /// ### Params
    ///
    /// * `a` - The A matrix
    /// * `b` - The B matrix
    ///
    /// ### Returns
    ///
    /// The residual sum of squares (RSS)
    fn compute_rss_trace(
        &self,
        a: &CompressedSparseData2<f32>,
        b: &CompressedSparseData2<f32>,
    ) -> Result<f32, BixverseErrors> {
        // Term 1: ||K||_F^2 (cached)
        let k_frob_sq = self.k_frobenius_norm_sq.unwrap();

        // K^2 @ B = K @ (K @ B)  [n × k]
        let k2_b = self.k_squared_matmul(b)?;

        // Term 2: -2 * trace(K^2 @ B @ A)
        // Reorder via cyclic property: trace(A @ K^2 @ B)  [k × k]
        let a_k2b = csr_matmul_csr(a, &k2_b);
        let trace_term = matrix_trace(&a_k2b);

        // Term 3: trace(A^T @ B^T @ K^2 @ B @ A)
        // Reorder via cyclic property: trace(A @ A^T @ B^T @ K^2 @ B)
        let a_t = a.transpose_and_convert();
        let a_at = csr_matmul_csr(a, &a_t); // [k × k]

        let b_t = b.transpose_and_convert();
        let bt_k2b = csr_matmul_csr(&b_t, &k2_b); // [k × k]

        let result = csr_matmul_csr(&a_at, &bt_k2b); // [k × k]
        let reconstruction_frob_sq = matrix_trace(&result);

        Ok((k_frob_sq - 2.0 * trace_term + reconstruction_frob_sq).sqrt())
    }

    /// Get hard cell assignments (each cell assigned to one SEACell)
    ///
    /// Transposes A to CSC for O(nnz) lookup per cell rather than the
    /// O(n × k × avg_nnz) linear scan used when iterating over CSR rows.
    ///
    /// ### Returns
    ///
    /// Vector of SEACell assignments (0 to k-1)
    pub fn get_hard_assignments(&self) -> Result<Vec<usize>, BixverseErrors> {
        if self.a.is_none() {
            return Err(BixverseErrors::SEACellsModelNotFitted);
        }

        let a = self.a.as_ref().unwrap();
        let n = a.shape.1;

        // A is (k × n) CSR. Transposing gives (n × k) CSR, equivalent to
        // (k × n) CSC, so each row corresponds to one cell with contiguous
        // entries over archetypes.
        let a_csc = a.transpose_and_convert();

        let mut assignments = vec![0usize; n];

        for cell in 0..n {
            let start = a_csc.indptr[cell];
            let end = a_csc.indptr[cell + 1];

            let mut max_val = f32::NEG_INFINITY;
            let mut max_arch = 0;
            for idx in start..end {
                if a_csc.data[idx] > max_val {
                    max_val = a_csc.data[idx];
                    max_arch = a_csc.indices[idx];
                }
            }
            assignments[cell] = max_arch;
        }

        Ok(assignments)
    }

    /// Get RSS history
    ///
    /// ### Returns
    ///
    /// Slice of RSS values recorded at each iteration
    pub fn get_rss_history(&self) -> &[f32] {
        &self.rss_history
    }

    /// Get the archetype cell indices
    ///
    /// ### Returns
    ///
    /// Vector of cell indices selected as archetypes
    ///
    /// ### Panics
    ///
    /// Panics if archetypes have not been initialised yet
    pub fn get_archetypes(&self) -> Result<Vec<usize>, BixverseErrors> {
        if self.archetypes.is_none() {
            return Err(BixverseErrors::SEACellsArchetypesMissing);
        }

        Ok(self.archetypes.as_ref().unwrap().clone())
    }
}
