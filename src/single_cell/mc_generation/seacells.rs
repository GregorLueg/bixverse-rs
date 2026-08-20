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
use crate::utils::simd::argmin_diff_simd_f32;

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
// Consts //
////////////

/// Divisor behind the adaptive bandwidth's neighbour rank.
///
/// The reference's `adaptive_k = floor(n_neighbors / 3)`, indexed one back. A
/// third of the neighbourhood is far enough out to be a stable local scale and
/// close enough in to still be local.
const BANDWIDTH_RANK_DIVISOR: usize = 3;

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
    ///
    /// Leave this on. The Frank-Wolfe updates only ever add atoms: `A` grows by
    /// one atom per iteration and `B` gains up to `k` entries per iteration via
    /// [sparse_add_csr], and `B` carries over into the next outer iteration.
    /// Without pruning the nnz of both climbs monotonically across the whole
    /// fit, and with it the cost of every `K^2 X` product, so the time per
    /// iteration keeps rising instead of settling.
    ///
    /// The accuracy cost is set by [SEACellsParams::pruning_threshold], not by
    /// this flag. With pruning off the atom weights collapse to
    /// `2(t + 1) / (T(T + 1))` for `T = max_fw_iters`, so any threshold below
    /// that floor removes numerical dust only.
    pub pruning: bool,
    /// Pruning threshold to apply. Choose it below the smallest weight the
    /// Frank-Wolfe schedule produces, `2 / (T(T + 1))` for `T = max_fw_iters`;
    /// above that, pruning starts removing live mass and shifts the solution.
    pub pruning_threshold: f32,
    /// Optional number of landmarks. If provided, it will use the Nystroem
    /// approach during archetype generation.
    pub n_landmarks: Option<usize>,
    // -- knn --
    /// [KnnParams] for the various approximate nearest neighbour searches
    /// in ann-search-rs
    pub knn_params: KnnParams,
    // -- eigensolver --
    /// [LanczosParams] for the diffusion-map eigendecomposition.
    pub lanczos_params: LanczosParams,
}

impl SEACellsParams {
    /// Generate a version of this with sensible base parameters
    ///
    /// The iteration counts follow the reference Python implementation of
    /// Persad, et al. `n_sea_cells` has no meaningful default and is left at
    /// zero, so callers have to set it.
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn new() -> Self {
        Self {
            // sea cells
            n_sea_cells: 0,
            max_fw_iters: 50,
            convergence_epsilon: 1e-3,
            max_iter: 100,
            min_iter: 10,
            greedy_threshold: 20000,
            graph_building: "union".to_string(),
            pruning: true,
            pruning_threshold: 1e-7,
            n_landmarks: None,
            // knn
            knn_params: KnnParams::default(),
            // eigensolver
            lanczos_params: LanczosParams::default(),
        }
    }
}

/// Default implementation for SEACellsParams
impl Default for SEACellsParams {
    fn default() -> Self {
        Self::new()
    }
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
/// Superseded on the live path by [prune_and_renormalise_tracked], which does
/// the same thing and additionally reports what it changed so `K² B` can be
/// mirrored. Retained for the iteration-major reference implementation.
///
/// ### Params
///
/// * `mat` - Mutable reference to the CompressedSparseData2 to be pruned
/// * `threshold` - Pruning threshold
///
/// ### Returns
///
/// Pruned matrix.
#[cfg(test)]
fn prune_and_renormalise(mat: &mut CompressedSparseData2<f32>, threshold: f32) {
    // remove values below threshold
    let mut new_data = Vec::new();
    let mut new_indices = Vec::new();
    let mut new_indptr = vec![0];

    for row in 0..mat.shape.0 {
        let start = mat.indptr[row] as usize;
        let end = mat.indptr[row + 1] as usize;

        for idx in start..end {
            if mat.data[idx].abs() > threshold {
                new_data.push(mat.data[idx]);
                new_indices.push(mat.indices[idx]);
            }
        }
        new_indptr.push(new_data.len() as u32);
    }

    mat.data = new_data;
    mat.indices = new_indices;
    mat.indptr = new_indptr;

    // renormalise columns to maintain sum-to-1 constraint
    normalise_csr_columns_l1(mat);
}

/// Compute the trace (sum of diagonal elements) of a sparse matrix
///
/// Accumulates in `f64`. The traces feed [SEACells::compute_rss_trace], where
/// they cancel against `||K||_F^2`, so the summation itself must not be the
/// thing that loses the answer.
///
/// ### Params
///
/// * `mat` - Sparse CSR matrix
///
/// ### Returns
///
/// Sum of diagonal elements `mat[i, i]`
fn matrix_trace(mat: &CompressedSparseData2<f32>) -> f64 {
    let n = mat.shape.0.min(mat.shape.1);
    let mut trace = 0.0f64;

    for i in 0..n {
        let row_start = mat.indptr[i] as usize;
        let row_end = mat.indptr[i + 1] as usize;

        for idx in row_start..row_end {
            if mat.indices[idx] == i as u32 {
                trace += mat.data[idx] as f64;
                break;
            }
        }
    }

    trace
}

/// Compute adaptive anisotropic diffusion kernel
///
/// Implementation from palantir package.  Uses the k/3-th nearest neighbour
/// distance as adaptive bandwidth. For edge (i,j) with distance d: weight =
/// exp(-d/σᵢ)
///
/// The bandwidth is derived per row from the neighbours actually supplied,
/// which is the same set the weights are computed over. Taking it from a
/// separately requested `k` breaks that invariant: it indexes out of bounds
/// when the supplied graph is narrower, and picks a bandwidth from the wrong
/// part of the neighbourhood when it is wider. The consequence worth knowing
/// about is that the *caller's* graph width sets the kernel bandwidth, so a
/// pipeline whose `params.knn` disagrees with the width of the `knn_indices` it
/// hands over gets the width, silently.
///
/// The rank is `BANDWIDTH_RANK_DIVISOR` into a **self-inclusive** count. The
/// reference asks scanpy for `n_neighbors = 30`, gets 29 stored distances back
/// with the self hit excluded, and indexes rank `floor(30 / 3) - 1 = 9` into
/// them. Deriving the rank from the 29 instead lands on rank 8, one neighbour
/// tighter, and every kernel bandwidth comes out slightly smaller.
///
/// A row of duplicate cells gives a zero bandwidth, and the matching zero
/// distance then turns the weight into `0 / 0 = NaN`. Zero and non-finite
/// bandwidths are replaced by the smallest positive one in the set, mirroring
/// [crate::single_cell::sc_trajectory::markov::build_waypoint_transitions]; the
/// reference divides by zero and silently kills the row.
///
/// ### Params
///
/// * `knn_indices` - kNN indices for each cell
/// * `knn_distances` - kNN distances for each cell
/// * `squared_dist` - Are the distances squared (squared Euclidean for
///   example).
///
/// ### Returns
///
/// Symmetric kernel matrix, or an error when a cell has no neighbours or the
/// kernel comes out non-finite.
pub fn compute_diffusion_kernel(
    knn_indices: &[Vec<usize>],
    knn_distances: &[Vec<f32>],
    squared_dist: bool,
) -> Result<CompressedSparseData2<f32>, BixverseErrors> {
    let n = knn_indices.len();

    let mut adaptive_std: Vec<f32> = knn_distances
        .iter()
        .enumerate()
        .map(|(i, dists)| {
            let mut sorted = dists.clone();
            // total_cmp rather than partial_cmp: a NaN distance in a
            // caller-supplied graph would otherwise panic the sort.
            sorted.sort_by(|a, b| a.total_cmp(b));
            // `+ 1` puts the count back on the reference's self-inclusive
            // footing; the supplied row has the self hit removed.
            sorted
                .get(((dists.len() + 1) / BANDWIDTH_RANK_DIVISOR).max(1) - 1)
                .copied()
                .ok_or_else(|| {
                    BixverseErrors::InvalidArgument(format!(
                        "kNN row {i} has no neighbours; cannot derive an adaptive bandwidth"
                    ))
                })
        })
        .collect::<Result<Vec<f32>, BixverseErrors>>()?;

    let smallest_positive = adaptive_std
        .iter()
        .copied()
        .filter(|s| *s > 0.0)
        .fold(f32::INFINITY, f32::min);
    let fallback = if smallest_positive.is_finite() {
        smallest_positive
    } else {
        1.0
    };
    for s in adaptive_std.iter_mut() {
        if !s.is_finite() || *s <= 0.0 {
            *s = fallback;
        }
    }

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

    let w = coo_to_csr(&rows.index_cast(), &cols.index_cast(), &vals, (n, n));

    // symmetrise: kernel = W + W^T
    let w_t = w.transpose_and_convert();

    let res = sparse_add_csr(&w, &w_t)?;

    // A NaN here survives everything downstream: `f32::min` / `f32::max` swallow
    // it in the min-max scaling, every NaN comparison in the boundary scan is
    // false, and it finally lands on a `partial_cmp().unwrap()` in the waypoint
    // sampler. Catch it where it is still attributable.
    if res.data.par_iter().any(|v| !v.is_finite()) {
        return Err(BixverseErrors::DiffusionKernelNotFinite { n_cells: n });
    }

    Ok(res)
}

/// Compute diffusion maps from kernel matrix
///
/// Normalises kernel to transition matrix and performs eigendecomposition.
///
/// ### Params
///
/// * `kernel` - Symmetric kernel matrix, normalised in place
/// * `n_components` - Number of eigenvectors to compute
/// * `seed` - Random seed for the Lanczos start vector
/// * `lanczos_params` - Optional [LanczosParams] for the eigensolver, defaulted
///   when `None`. A diffusion kernel over a long, thin manifold has a tightly
///   clustered spectrum, so the restart budget is what decides whether the
///   components come back resolved.
///
/// ### Returns
///
/// `(eigenvalues, eigenvectors)`, the eigenvectors `n` by the returned
/// eigenvalue count. That count is `min(n_components, n)` and can be smaller
/// still when the solver hits an invariant subspace, so read it off the result
/// rather than assuming `n_components`. Eigenvalues are `f64` because the
/// multiscale scaling forms `lambda / (1 - lambda)` and `lambda` sits within a
/// few `f32` ulps of one for any well-connected kernel.
pub fn diffusion_map_from_kernel(
    kernel: &mut CompressedSparseData2<f32>,
    n_components: usize,
    seed: u64,
    lanczos_params: Option<LanczosParams>,
) -> Result<(Vec<f64>, Vec<Vec<f32>>), BixverseErrors> {
    let res = diffusion_map_from_kernel_diag(kernel, n_components, seed, lanczos_params)?;
    Ok((res.eigenvalues, res.eigenvectors))
}

/// Compute diffusion maps from a kernel matrix, keeping the solver diagnostics
///
/// Same normalisation and solve as [diffusion_map_from_kernel]. A trajectory is
/// a long, thin manifold, which is the worst case for the eigensolver, and a
/// budget-exhausted solve is otherwise indistinguishable from a converged one:
/// the embedding comes back as noise and every result downstream still looks
/// perfectly well formed. Callers that report to a user should take this one.
///
/// ### Params
///
/// * `kernel` - Symmetric kernel matrix, normalised in place
/// * `n_components` - Number of eigenvectors to compute
/// * `seed` - Random seed for the Lanczos start vector
/// * `lanczos_params` - Optional [LanczosParams] for the eigensolver, defaulted
///   when `None`.
///
/// ### Returns
///
/// The [LanczosResult], carrying the eigenpairs alongside `converged`,
/// `residual`, `norm_estimate` and `restarts`.
pub fn diffusion_map_from_kernel_diag(
    kernel: &mut CompressedSparseData2<f32>,
    n_components: usize,
    seed: u64,
    lanczos_params: Option<LanczosParams>,
) -> Result<LanczosResult, BixverseErrors> {
    // Compute row sums (degrees)
    let row_sums: Vec<f32> = (0..kernel.shape.0)
        .map(|i| {
            (kernel.indptr[i]..kernel.indptr[i + 1])
                .map(|idx| kernel.data[idx as usize])
                .sum()
        })
        .collect();

    // symmetric normalisation: D^(-1/2) * K * D^(-1/2)
    for i in 0..kernel.shape.0 {
        let d_i_sqrt = row_sums[i].sqrt();
        for idx in kernel.indptr[i]..kernel.indptr[i + 1] {
            let j = kernel.indices[idx as usize] as usize;
            let d_j_sqrt = row_sums[j].sqrt();
            kernel.data[idx as usize] /= d_i_sqrt * d_j_sqrt;
        }
    }

    compute_largest_eigenpairs_lanczos_diag(kernel, n_components, seed, lanczos_params)
}

/// Largest eigenvalue admitted into the `lambda / (1 - lambda)` scaling.
///
/// A diffusion operator has `lambda_0 = 1` exactly and, on a nearly
/// disconnected graph, several more within rounding of it. The unclamped scale
/// then overflows and poisons every downstream distance. Clamping at
/// `1 - 1e-10` caps the scale at `1e10`, orders of magnitude beyond any real
/// diffusion component, so genuine signal is untouched.
const MAX_MULTISCALE_LAMBDA: f64 = 1.0 - 1e-10;

/// Determine multiscale space by scaling eigenvectors
///
/// Scales eigenvectors by λᵢ/(1-λᵢ) for diffusion distance metric. The scaling
/// is done in `f64` and the eigenvalue clamped below one first; `1 - lambda` is
/// a subtraction of nearly equal numbers and loses every significant digit in
/// `f32` for the leading components.
///
/// ### Params
///
/// * `eigenvalues` - Eigenvalues from diffusion maps
/// * `eigenvectors` - Eigenvectors (n × components), the column count matching
///   `eigenvalues`
/// * `n_eigs` - Optional number of eigenvectors to use (None = auto-detect via
///   eigengap). Silently capped at the number actually available, because the
///   eigensolver can return fewer pairs than were asked for.
///
/// ### Returns
///
/// Scaled eigenvectors, `n` by `max(0, used - 1)`, the trivial leading
/// eigenvector dropped. Empty rows when fewer than two eigenvalues are
/// available.
pub fn determine_multiscale_space(
    eigenvalues: &[f64],
    eigenvectors: &[Vec<f32>],
    n_eigs: Option<usize>,
) -> Vec<Vec<f32>> {
    let n = eigenvectors.len();
    let available = eigenvalues
        .len()
        .min(eigenvectors.first().map_or(0, Vec::len));

    // auto-detect n_eigs using eigengap if not provided
    let requested = if let Some(n) = n_eigs {
        n
    } else {
        let gaps: Vec<f64> = eigenvalues.windows(2).map(|w| w[0] - w[1]).collect();

        let max_gap_idx = gaps
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx + 1)
            .unwrap_or(3);

        max_gap_idx.max(3)
    };

    let use_n_eigs = requested.min(available);
    let use_indices: Vec<usize> = (1..use_n_eigs).collect();

    let mut scaled = vec![vec![0.0f32; use_indices.len()]; n];

    for (out_idx, &eig_idx) in use_indices.iter().enumerate() {
        let lambda = eigenvalues[eig_idx].min(MAX_MULTISCALE_LAMBDA);
        let scale = (lambda / (1.0 - lambda)) as f32;

        for i in 0..n {
            scaled[i][out_idx] = eigenvectors[i][eig_idx] * scale;
        }
    }

    scaled
}

/// Max-min waypoint sampling
///
/// For each dimension, iteratively selects points maximising the minimum
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
/// Indices of selected waypoints. Empty when there is nothing to sample over,
/// which happens when the diffusion map came back with fewer than two
/// components and the multiscale space is therefore zero-width.
pub fn max_min_sampling(data: &[Vec<f32>], num_waypoints: usize, seed: u64) -> Vec<usize> {
    let n = data.len();
    let n_dims = data.first().map_or(0, Vec::len);
    if n == 0 || n_dims == 0 {
        return Vec::new();
    }
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
                .map(|idx| kernel.data[idx as usize])
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
    coo_to_csr(&rows.index_cast(), &cols.index_cast(), &vals, (n, l))
}

/// Nystroem extension: y(i)[d] = (1/λ_d) · Σ_l P_nl[i,l] · y_landmark[l][d]
///
/// ### Params
///
/// * `p_nl` - Row-stochastic data-to-landmark transitions, `n × l`
/// * `landmark_embedding` - Multiscale embedding of the landmarks, `l × n_dim`
/// * `lambdas` - Eigenvalue per embedding dimension; a dimension with a
///   near-zero eigenvalue is left unscaled rather than divided by it
///
/// ### Returns
///
/// The embedding extended to all `n` cells, `n × n_dim`.
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
            let start = p_nl.indptr[i] as usize;
            let end = p_nl.indptr[i + 1] as usize;
            for idx in start..end {
                let l = p_nl.indices[idx] as usize;
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

////////////////////////
// Frank-Wolfe column //
////////////////////////

/// Below this the L1 renormalisation is skipped, matching
/// `normalise_csr_columns_l1`, which leaves a column alone when its sum does
/// not exceed this. Reproducing the guard exactly is what keeps a fully-pruned
/// column behaving identically on both paths.
const FW_RENORM_FLOOR: f64 = 1e-15;

/// Atom bookkeeping for one Frank-Wolfe column.
///
/// A column of `A` (or `B`) is a convex combination of at most `max_fw_iters`
/// one-hot atoms. Rather than rebuilding the sparse matrix each iteration, this
/// tracks the `(index, weight)` pairs and reports what each operation changed,
/// so the caller can apply the matching correction to whatever gradient state
/// it maintains:
///
/// - the convex step scales every weight by `1 - γ` and adds `γ` to one atom,
///   so the gradient state scales and takes one rank-1 update;
/// - pruning removes atoms outright, so the gradient state takes one rank-1
///   *subtraction* per dropped atom, which can happen at most once per atom;
/// - renormalisation scales every weight, so the gradient state scales.
///
/// The whole point is that none of these needs the gradient recomputed from
/// scratch, which is what the iteration-major formulation does.
#[derive(Clone, Debug, Default)]
pub struct FwAtoms {
    /// Atom indices, unique and in insertion order
    indices: Vec<u32>,
    /// Atom weights, parallel to `indices`
    weights: Vec<f32>,
}

/// What a `prune` call removed, so the caller can correct its gradient state.
#[derive(Clone, Debug, Default)]
pub struct FwPruneOutcome {
    /// Dropped `(index, weight)` pairs, weights as they stood *before* the
    /// renormalisation
    pub dropped: Vec<(u32, f32)>,
    /// Factor the surviving weights were multiplied by. `1.0` when nothing
    /// needed renormalising.
    pub renorm: f32,
}

impl FwAtoms {
    /// Empty atom set with capacity for `max_atoms`
    ///
    /// ### Params
    ///
    /// * `max_atoms` - Upper bound on distinct atoms, i.e. `max_fw_iters`
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn with_capacity(max_atoms: usize) -> Self {
        Self {
            indices: Vec::with_capacity(max_atoms),
            weights: Vec::with_capacity(max_atoms),
        }
    }

    /// Seed from an existing sparse column
    ///
    /// ### Params
    ///
    /// * `indices` - Atom indices
    /// * `weights` - Atom weights
    pub fn reset_from(&mut self, indices: &[u32], weights: &[f32]) {
        self.indices.clear();
        self.weights.clear();
        self.indices.extend_from_slice(indices);
        self.weights.extend_from_slice(weights);
    }

    /// Clear all atoms
    pub fn clear(&mut self) {
        self.indices.clear();
        self.weights.clear();
    }

    /// The current atoms
    ///
    /// ### Returns
    ///
    /// `(indices, weights)`, parallel slices.
    pub fn atoms(&self) -> (&[u32], &[f32]) {
        (&self.indices, &self.weights)
    }

    /// Number of atoms currently held
    ///
    /// ### Returns
    ///
    /// The atom count.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Whether the column is empty
    ///
    /// ### Returns
    ///
    /// `true` if no atoms are held.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Apply the Frank-Wolfe convex step `X ← (1 - γ)X + γ e_amin`
    ///
    /// A repeated `amin` merges into the existing atom rather than creating a
    /// duplicate, matching what `sparse_add_csr` does on the iteration-major
    /// path.
    ///
    /// ### Params
    ///
    /// * `gamma` - Step size
    /// * `amin` - Index of the atom receiving `gamma`
    pub fn step(&mut self, gamma: f32, amin: u32) {
        let retain = 1.0 - gamma;
        for weight in &mut self.weights {
            *weight *= retain;
        }
        match self.indices.iter().position(|&i| i == amin) {
            Some(pos) => self.weights[pos] += gamma,
            None => {
                self.indices.push(amin);
                self.weights.push(gamma);
            }
        }
    }

    /// Drop atoms at or below `threshold`, then renormalise the survivors to
    /// sum to 1
    ///
    /// The keep test is `|w| > threshold` and the renormalisation is skipped
    /// for a surviving mass at or below `FW_RENORM_FLOOR`, both matching
    /// `prune_and_renormalise` and `normalise_csr_columns_l1`.
    ///
    /// ### Params
    ///
    /// * `threshold` - Pruning threshold
    ///
    /// ### Returns
    ///
    /// The [FwPruneOutcome] describing what to correct in the gradient state.
    pub fn prune(&mut self, threshold: f32) -> FwPruneOutcome {
        let mut dropped = Vec::new();
        let mut write = 0;
        for read in 0..self.indices.len() {
            if self.weights[read].abs() > threshold {
                self.indices[write] = self.indices[read];
                self.weights[write] = self.weights[read];
                write += 1;
            } else {
                dropped.push((self.indices[read], self.weights[read]));
            }
        }
        self.indices.truncate(write);
        self.weights.truncate(write);

        let sum: f32 = self.weights.iter().sum();
        let renorm = if (sum as f64) > FW_RENORM_FLOOR {
            1.0 / sum
        } else {
            1.0
        };
        if renorm != 1.0 {
            for weight in &mut self.weights {
                *weight *= renorm;
            }
        }

        FwPruneOutcome { dropped, renorm }
    }
}

///////////////////////
// B argmin back end //
///////////////////////

/// Back end for the Frank-Wolfe inner solves, on whatever device.
///
/// This is the seam between the CPU and GPU paths: the GPU entry point swaps it
/// out and reuses the rest of the loop rather than forking it. Despite the
/// name, which is kept because the trait is public and implemented downstream,
/// it now covers both Frank-Wolfe solves.
///
/// [begin] is called once per B update, when `t1` and `t2` change; [argmins]
/// once per Frank-Wolfe iteration, when `K²B` and `B` change. Splitting them
/// keeps the per-update uploads out of the per-iteration path. [columns_a] is
/// called once per A update and defaults to the CPU implementation, so an
/// existing implementor keeps working untouched.
///
/// Neither solve on its own bounds the fit: sampled at 50k cells and 666
/// archetypes, [argmins] was about a quarter of wall-clock and [columns_a]
/// another quarter. That split predates the current kernels and both have since
/// got faster, so treat it as the reason the seam covers both rather than as a
/// current attribution.
///
/// [begin]: FwArgminB::begin
/// [argmins]: FwArgminB::argmins
/// [columns_a]: FwArgminB::columns_a
pub trait FwArgminB {
    /// Bind the terms that stay fixed across one B update.
    ///
    /// ### Params
    ///
    /// * `t1` - k × k matrix A Aᵀ (symmetric)
    /// * `t2` - n × k matrix K² Aᵀ
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or a back end specific error.
    fn begin(
        &mut self,
        t1: &CompressedSparseData2<f32>,
        t2: &CompressedSparseData2<f32>,
    ) -> Result<(), BixverseErrors>;

    /// Argmin per archetype and the Frank-Wolfe duality gap.
    ///
    /// ### Params
    ///
    /// * `k2_b` - n × k matrix K² B for the current B
    /// * `b` - n × k current B
    ///
    /// ### Returns
    ///
    /// The argmin cell index per archetype, and the absolute duality gap.
    fn argmins(
        &mut self,
        k2_b: &CompressedSparseData2<f32>,
        b: &CompressedSparseData2<f32>,
    ) -> Result<(Vec<usize>, f32), BixverseErrors>;

    /// Frank-Wolfe column solve for the A update, one column per cell.
    ///
    /// Defaults to the crate-internal `fw_columns_a`, so a back end that only
    /// accelerates the B argmin needs no change.
    ///
    /// ### Params
    ///
    /// * `t1` - k × k matrix Bᵀ K² B
    /// * `a_prev_t` - n × k, rows are columns of the previous A
    /// * `k2_b` - n × k matrix K² B
    /// * `k` - Number of archetypes
    /// * `n` - Number of cells
    /// * `n_iters` - Frank-Wolfe iterations per column
    /// * `pruning` - Pruning threshold, or `None` to skip pruning
    ///
    /// ### Returns
    ///
    /// One [FwAtoms] per cell, in cell order.
    #[allow(clippy::too_many_arguments)]
    fn columns_a(
        &mut self,
        t1: &CompressedSparseData2<f32>,
        a_prev_t: &CompressedSparseData2<f32>,
        k2_b: &CompressedSparseData2<f32>,
        k: usize,
        n: usize,
        n_iters: usize,
        pruning: Option<f32>,
    ) -> Result<Vec<FwAtoms>, BixverseErrors> {
        Ok(fw_columns_a(t1, a_prev_t, k2_b, k, n, n_iters, pruning))
    }
}

/// CPU back end, wrapping `fw_argmins_b`.
///
/// Holds the transposes the scan needs. `fw_argmins_b` walks columns of `K²B`,
/// `K²Aᵀ` and `B`, so all three arrive here untransposed and are converted on
/// the way in, exactly as the pre-seam `update_b_mat` did.
#[derive(Default)]
pub struct CpuFwArgminB {
    /// k × k matrix A Aᵀ
    t1: Option<CompressedSparseData2<f32>>,
    /// k × n, rows are columns of K² Aᵀ
    t2_t: Option<CompressedSparseData2<f32>>,
}

impl FwArgminB for CpuFwArgminB {
    fn begin(
        &mut self,
        t1: &CompressedSparseData2<f32>,
        t2: &CompressedSparseData2<f32>,
    ) -> Result<(), BixverseErrors> {
        self.t1 = Some(t1.clone());
        self.t2_t = Some(t2.transpose_and_convert());
        Ok(())
    }

    fn argmins(
        &mut self,
        k2_b: &CompressedSparseData2<f32>,
        b: &CompressedSparseData2<f32>,
    ) -> Result<(Vec<usize>, f32), BixverseErrors> {
        let t1 = self
            .t1
            .as_ref()
            .ok_or(BixverseErrors::SEACellsModelNotFitted)?;
        let t2_t = self
            .t2_t
            .as_ref()
            .ok_or(BixverseErrors::SEACellsModelNotFitted)?;

        let (n, k) = (k2_b.shape.0, k2_b.shape.1);
        let k2_b_t = k2_b.transpose_and_convert();
        let b_t = b.transpose_and_convert();

        Ok(fw_argmins_b(&k2_b_t, t1, t2_t, &b_t, n, k))
    }
}

////////////////////
// Matrix updates //
////////////////////

/////////
// Old //
/////////

/// Per-cell Frank-Wolfe argmins for the A update, one gradient column at a
/// time so the full k × n gradient is never materialised. The factor of 2 in
/// the true gradient is dropped: irrelevant to an argmin.
///
/// Superseded by [fw_columns_a], which runs all Frank-Wolfe iterations for a
/// cell in one pass. Retained as the reference the rewrite is checked against.
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
#[cfg(test)]
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

                    for ai in a_t.indptr[j] as usize..a_t.indptr[j + 1] as usize {
                        let r = a_t.indices[ai] as usize;
                        let w = a_t.data[ai];
                        for ti in t1.indptr[r] as usize..t1.indptr[r + 1] as usize {
                            buf[t1.indices[ti] as usize] += w * t1.data[ti];
                        }
                    }
                    for ki in k2_b.indptr[j] as usize..k2_b.indptr[j + 1] as usize {
                        buf[k2_b.indices[ki] as usize] -= k2_b.data[ki];
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

/// Cell-major Frank-Wolfe pass for the A update.
///
/// ### Params
///
/// * `t1` - k × k matrix Bᵀ K² B (symmetric)
/// * `a_prev_t` - n × k, row `j` is column `j` of the previous A
/// * `k2_b` - n × k matrix K² B
/// * `k` - Number of archetypes
/// * `n` - Number of cells
/// * `n_iters` - Frank-Wolfe iterations to run
/// * `pruning` - Pruning threshold, or `None` to skip pruning
///
/// ### Returns
///
/// One [FwAtoms] per cell, holding that column of the updated A.
pub(crate) fn fw_columns_a(
    t1: &CompressedSparseData2<f32>,
    a_prev_t: &CompressedSparseData2<f32>,
    k2_b: &CompressedSparseData2<f32>,
    k: usize,
    n: usize,
    n_iters: usize,
    pruning: Option<f32>,
) -> Vec<FwAtoms> {
    /// Cells per rayon task. Large enough that the two `k`-length scratch
    /// buffers are reused many times per task, small enough to keep the tail
    /// balanced.
    const CHUNK: usize = 256;

    let n_chunks = n.div_ceil(CHUNK);

    let chunks: Vec<Vec<FwAtoms>> = (0..n_chunks)
        .into_par_iter()
        .map_init(
            || (vec![0.0f32; k], vec![0.0f32; k]),
            |(w, k2b_row), chunk_idx| {
                let start = chunk_idx * CHUNK;
                let end = ((chunk_idx + 1) * CHUNK).min(n);
                let mut out = Vec::with_capacity(end - start);

                for j in start..end {
                    // stage the gradient's constant term, -K²B[j, :]
                    k2b_row.fill(0.0);
                    for ki in k2_b.indptr[j] as usize..k2_b.indptr[j + 1] as usize {
                        k2b_row[k2_b.indices[ki] as usize] = k2_b.data[ki];
                    }

                    // seed the atoms and w = t1 · A_prev[:, j]
                    let seed = a_prev_t.indptr[j] as usize..a_prev_t.indptr[j + 1] as usize;
                    let mut atoms = FwAtoms::with_capacity(n_iters + seed.len());
                    atoms.reset_from(
                        &a_prev_t.indices[seed.clone()],
                        &a_prev_t.data[seed.clone()],
                    );

                    w.fill(0.0);
                    for ai in seed {
                        let r = a_prev_t.indices[ai] as usize;
                        let weight = a_prev_t.data[ai];
                        for ti in t1.indptr[r] as usize..t1.indptr[r + 1] as usize {
                            w[t1.indices[ti] as usize] += weight * t1.data[ti];
                        }
                    }

                    for t in 0..n_iters {
                        // fused argmin over the gradient, which is never stored
                        let (amin, _) = argmin_diff_simd_f32(&w[..k], &k2b_row[..k]);

                        let gamma = 2.0 / (t as f32 + 2.0);
                        atoms.step(gamma, amin as u32);

                        let retain = 1.0 - gamma;
                        for value in w.iter_mut() {
                            *value *= retain;
                        }
                        for ti in t1.indptr[amin] as usize..t1.indptr[amin + 1] as usize {
                            w[t1.indices[ti] as usize] += gamma * t1.data[ti];
                        }

                        if let Some(threshold) = pruning {
                            let outcome = atoms.prune(threshold);
                            for (r, weight) in outcome.dropped {
                                let r = r as usize;
                                for ti in t1.indptr[r] as usize..t1.indptr[r + 1] as usize {
                                    w[t1.indices[ti] as usize] -= weight * t1.data[ti];
                                }
                            }
                            if outcome.renorm != 1.0 {
                                for value in w.iter_mut() {
                                    *value *= outcome.renorm;
                                }
                            }
                        }
                    }

                    out.push(atoms);
                }
                out
            },
        )
        .collect();

    chunks.into_iter().flatten().collect()
}

/// Assemble a sparse matrix from per-column Frank-Wolfe atom lists.
///
/// ### Params
///
/// * `columns` - One [FwAtoms] per column, in column order
/// * `shape` - `(nrow, ncol)` of the result
///
/// ### Returns
///
/// The matrix in CSR, with column indices sorted within each row.
fn fw_atoms_to_csr(columns: &[FwAtoms], shape: (usize, usize)) -> CompressedSparseData2<f32> {
    let nnz: usize = columns.iter().map(|c| c.len()).sum();
    let mut rows = Vec::with_capacity(nnz);
    let mut cols = Vec::with_capacity(nnz);
    let mut vals = Vec::with_capacity(nnz);

    for (col, atoms) in columns.iter().enumerate() {
        let (indices, weights) = atoms.atoms();
        for (&row, &weight) in indices.iter().zip(weights.iter()) {
            rows.push(row as usize);
            cols.push(col);
            vals.push(weight);
        }
    }

    coo_to_csr(&rows.index_cast(), &cols.index_cast(), &vals, shape)
}

/// Per-archetype Frank-Wolfe argmins and FW duality gap for the B update, one
/// gradient column at a time. Factor of 2 dropped: cancels in the gap ratio,
/// irrelevant to the argmin.
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
                    for ti in t1.indptr[c] as usize..t1.indptr[c + 1] as usize {
                        let m = t1.indices[ti] as usize;
                        let tmc = t1.data[ti];
                        for ki in k2_b_t.indptr[m] as usize..k2_b_t.indptr[m + 1] as usize {
                            buf[k2_b_t.indices[ki] as usize] += tmc * k2_b_t.data[ki];
                        }
                    }
                    for si in t2_t.indptr[c] as usize..t2_t.indptr[c + 1] as usize {
                        buf[t2_t.indices[si] as usize] -= t2_t.data[si];
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

                    for bi in b_t.indptr[c] as usize..b_t.indptr[c + 1] as usize {
                        g_dot_b += b_t.data[bi] * buf[b_t.indices[bi] as usize];
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

///////////////////////
// K^2 B maintenance //
///////////////////////

/// Frank-Wolfe iterations between full recomputes of `K² B`.
///
/// The incremental update is exact, so this is not a correctness backstop for
/// the arithmetic. It bounds fp drift and caps the sparsity pattern in the
/// worst case.
///
/// It has to sit below `MIN_FW_ITERS` in [SEACells::update_b_mat] to fire at
/// all: the loop may break as soon as `t >= MIN_FW_ITERS`, so a larger interval
/// simply never runs on a B update that converges early, which is the common
/// case.
///
/// The pattern does not in fact run away without it. `sparse_add_csr` drops
/// overlaps below its own epsilon, which removes exactly the entries the prune
/// corrections cancel, and those are always present in both addends. That is a
/// property of a shared helper rather than of anything here, so the refresh
/// stays as the thing this module actually controls.
const K2B_REFRESH_EVERY: usize = 8;

/// Frank-Wolfe iterations the B update always runs before it may converge.
///
/// Module scope rather than local because [K2B_REFRESH_EVERY] is only reachable
/// while this holds: the loop cannot break earlier, so an interval below this
/// always fires at least once, and one above it never fires on a B update that
/// converges early.
const MIN_FW_ITERS: usize = 10;

const _: () = assert!(
    K2B_REFRESH_EVERY < MIN_FW_ITERS,
    "K2B_REFRESH_EVERY must sit below MIN_FW_ITERS or the refresh never fires"
);

/// Accumulate weighted columns of `K²` into a CSC matrix.
///
/// `K` is symmetric, so `K²[:, j] == K²[j, :]`, and row `j` of `K²` is the
/// weighted merge of the `K` rows named by row `j` of `K`. Everything therefore
/// reads `K`'s CSR layout directly and no transpose or CSC of `K` is needed.
///
/// Each column gets a dense `n`-length accumulator with a stamp array, so the
/// per-column clear costs the number of touched entries rather than `n`.
/// Columns are processed in chunks because rayon calls a `map_init` initialiser
/// once per split job rather than once per thread, and allocating plus zeroing
/// two `n`-length buffers per column is far more work than the merge itself.
/// This matches [fw_columns_a] and [fw_argmins_b].
///
/// ### Params
///
/// * `kernel` - Symmetric kernel `K` as CSR `n × n`
/// * `per_column` - `per_column[c]` lists `(j, w)` pairs, meaning column `c` of
///   the result is `Σ w · K²[:, j]`. Empty lists give empty columns.
/// * `n` - Number of cells
///
/// ### Returns
///
/// The combination as CSC `n × k`. Row indices within a column are in scatter
/// order, **not** sorted: the only consumer is [transpose_sparse], which counts
/// per row and then scatters in column order, so its output rows come out
/// column-sorted whatever order the input column was in.
fn k_squared_column_combination(
    kernel: &CompressedSparseData2<f32>,
    per_column: &[Vec<(u32, f32)>],
    n: usize,
) -> CompressedSparseData2<f32> {
    const CHUNK: usize = 64;

    let k = per_column.len();

    let columns: Vec<(Vec<u32>, Vec<f32>)> = per_column
        .par_chunks(CHUNK)
        .map_init(
            || (vec![0.0f32; n], vec![0u32; n], Vec::<u32>::new(), 0u32),
            |(acc, stamp, touched, epoch), chunk| {
                let mut out = Vec::with_capacity(chunk.len());

                for sources in chunk.iter() {
                    *epoch += 1;
                    let tag = *epoch;
                    touched.clear();

                    for &(j, weight) in sources.iter() {
                        // A dropped atom that was already zero contributes
                        // nothing, and at `t = 0` every seed atom is one of
                        // those. Skipping them here avoids a full `K²` row merge
                        // per entry for a result the filter below discards.
                        if weight == 0.0 {
                            continue;
                        }
                        let j = j as usize;
                        for mid in kernel.indptr[j] as usize..kernel.indptr[j + 1] as usize {
                            let m = kernel.indices[mid] as usize;
                            let scaled = kernel.data[mid] * weight;
                            for oid in kernel.indptr[m] as usize..kernel.indptr[m + 1] as usize {
                                let o = kernel.indices[oid] as usize;
                                if stamp[o] != tag {
                                    stamp[o] = tag;
                                    acc[o] = 0.0;
                                    touched.push(o as u32);
                                }
                                acc[o] += scaled * kernel.data[oid];
                            }
                        }
                    }

                    let mut indices = Vec::with_capacity(touched.len());
                    let mut data = Vec::with_capacity(touched.len());
                    for &o in touched.iter() {
                        let value = acc[o as usize];
                        if value != 0.0 {
                            indices.push(o);
                            data.push(value);
                        }
                    }
                    out.push((indices, data));
                }

                out
            },
        )
        .flatten()
        .collect();

    let nnz: usize = columns.iter().map(|(idx, _)| idx.len()).sum();
    let mut indptr = Vec::with_capacity(k + 1);
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    indptr.push(0u32);
    for (col_indices, col_data) in columns {
        indices.extend_from_slice(&col_indices);
        data.extend_from_slice(&col_data);
        indptr.push(indices.len() as u32);
    }

    CompressedSparseData2 {
        data,
        indices,
        indptr,
        cs_type: CompressedSparseFormat::Csc,
        data_2: None,
        shape: (n, k),
    }
}

/// Prune tiny values and L1-normalise columns, reporting what changed.
///
/// Same operation as [prune_and_renormalise], but returns enough to mirror it
/// onto `K² B` exactly. Mirroring `B ← (B - D) diag(s)` needs both the dropped
/// entries `D` and the per-column factors `s`, since
/// `K²(B - D) diag(s) = (K²B - Σ v · K²[:, j]) diag(s)`.
///
/// ### Params
///
/// * `mat` - CSR matrix to prune in place
/// * `threshold` - Values at or below this magnitude are dropped
///
/// ### Returns
///
/// `(dropped, factors)` where `dropped` holds `(column, row, value)` for every
/// removed entry and `factors[c]` is the multiplier applied to column `c`.
fn prune_and_renormalise_tracked(
    mat: &mut CompressedSparseData2<f32>,
    threshold: f32,
) -> (Vec<(u32, u32, f32)>, Vec<f32>) {
    assert!(
        mat.cs_type.is_csr(),
        "prune_and_renormalise_tracked expects CSR; a CSC input would read rows as columns"
    );

    let ncols = mat.shape.1;
    let mut dropped = Vec::new();
    let mut new_data = Vec::with_capacity(mat.data.len());
    let mut new_indices = Vec::with_capacity(mat.indices.len());
    let mut new_indptr = Vec::with_capacity(mat.indptr.len());
    new_indptr.push(0u32);

    for row in 0..mat.shape.0 {
        for idx in mat.indptr[row] as usize..mat.indptr[row + 1] as usize {
            let value = mat.data[idx];
            let col = mat.indices[idx];
            if value.abs() > threshold {
                new_data.push(value);
                new_indices.push(col);
            } else {
                dropped.push((col, row as u32, value));
            }
        }
        new_indptr.push(new_data.len() as u32);
    }

    mat.data = new_data;
    mat.indices = new_indices;
    mat.indptr = new_indptr;

    let mut col_sums = vec![0.0f32; ncols];
    for (idx, &col) in mat.indices.iter().enumerate() {
        col_sums[col as usize] += mat.data[idx];
    }
    let factors: Vec<f32> = col_sums
        .iter()
        .map(|&sum| {
            if (sum as f64) > FW_RENORM_FLOOR {
                1.0 / sum
            } else {
                1.0
            }
        })
        .collect();
    for (idx, &col) in mat.indices.iter().enumerate() {
        mat.data[idx] *= factors[col as usize];
    }

    (dropped, factors)
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
/// - Never materialises `K²`. Products go through `K @ (K @ X)`, and the one
///   product the Frank-Wolfe loop needs every iteration, `K² B`, is cached and
///   updated incrementally from weighted columns of `K²`
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
    /// Cached `K² B` (n × k), maintained incrementally through the Frank-Wolfe
    /// B loop.
    ///
    /// Invariant: whenever this and [SEACells::b] are both `Some`, this equals
    /// `K² · b`. Every consumer of `K² B` reads it from here rather than
    /// recomputing, so anything that writes `b` must write this in the same
    /// step. [K2B_REFRESH_EVERY] governs how often it is rebuilt from scratch.
    k2_b: Option<CompressedSparseData2<f32>>,
    /// Indices of cells selected as initial archetypes.
    archetypes: Option<Vec<usize>>,
    /// Residual sum of squares at each iteration.
    rss_history: Vec<f32>,
    /// Absolute RSS change threshold for convergence.
    convergence_threshold: Option<f32>,
    ///  Cached ||K||_F^2 for trace-based RSS, accumulated in `f64`.
    k_frobenius_norm_sq: Option<f64>,
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
            k2_b: None,
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
    /// K² is never materialised - downstream operations either compute
    /// K @ (K @ X) or merge single weighted columns of K² out of K's rows,
    /// bounding memory to O(nnz(K)).
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

        let kernel = coo_to_csr(&rows.index_cast(), &cols.index_cast(), &vals, (n, n));

        if verbosity.detailed_verbosity() {
            println!(" Pre-computing kernel Frobenius norm...");
        }
        self.k_frobenius_norm_sq = Some(frobenius_norm_sq_f64(&kernel));

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
        let kx = csr_matmul_csr(k, x)?;
        let res = csr_matmul_csr(k, &kx)?;

        Ok(res)
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
    fn k_squared_matvec(&self, v: &[f32]) -> Result<Vec<f32>, BixverseErrors> {
        let k = self.kernel_mat.as_ref().unwrap();
        let kv = csr_matvec(k, v)?;
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
        self.fit_with(seed, verbose, &mut CpuFwArgminB::default())
    }

    /// Fit the SEACells model against a chosen B-argmin back end
    ///
    /// Same loop as [SEACells::fit], with the one step that dominates the
    /// runtime pluggable. The GPU entry point uses this rather than duplicating
    /// the outer loop; see [FwArgminB].
    ///
    /// ### Params
    ///
    /// * `seed` - Random seed for reproducibility
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    /// * `backend` - The gradient argmin implementation to use
    pub fn fit_with(
        &mut self,
        seed: usize,
        verbose: usize,
        backend: &mut impl FwArgminB,
    ) -> Result<(), BixverseErrors> {
        let verbosity = parse_verbosity_level(verbose);

        if self.kernel_mat.is_none() {
            return Err(BixverseErrors::SEACellsKernelMatrixMissing);
        }
        if self.archetypes.is_none() {
            return Err(BixverseErrors::SEACellsArchetypesMissing);
        }

        self.initialise_matrices(verbose, seed as u64, backend)?;

        let a = self.a.as_ref().unwrap();
        let b = self.b.as_ref().unwrap();
        let k2_b = self.k2_b.as_ref().unwrap();

        let initial_rss = self.compute_rss(a, b, k2_b)?;
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
            let k2_b_current = self.k2_b.take().unwrap();

            debug_assert!(
                self.k2_b_matches(&b_current, &k2_b_current),
                "the K^2 B cache drifted from B before iteration {}",
                n_iter
            );

            let a_new =
                self.update_a_mat(&b_current, &k2_b_current, &a_current, verbose, backend)?;
            let (b_new, k2_b_new) =
                self.update_b_mat(&a_new, &b_current, &k2_b_current, verbose, backend)?;

            debug_assert!(
                self.k2_b_matches(&b_new, &k2_b_new),
                "the K^2 B cache drifted from B after iteration {}",
                n_iter
            );

            let rss = self.compute_rss(&a_new, &b_new, &k2_b_new)?;
            self.rss_history.push(rss);

            self.a = Some(a_new);
            self.b = Some(b_new);
            self.k2_b = Some(k2_b_new);

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
    /// For large datasets (>= greedy_threshold): uses fast random
    /// initialisation
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

        let mut kernel = compute_diffusion_kernel(knn_indices, knn_distances, squared_dist)?;

        let (eigenvalues, eigenvectors) = diffusion_map_from_kernel(
            &mut kernel,
            self.params.knn_params.k,
            seed,
            Some(self.params.lanczos_params),
        )?;

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
        let greedy_ix = self.get_greedy_centres(from_greedy + 10)?;

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
        let kernel = compute_diffusion_kernel(knn_indices, knn_distances, squared_dist)?;

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

        let mut ll_kernel = compute_diffusion_kernel(&ll_idx, &ll_dist, squared_dist)?;

        let n_eigs = k_ll.min(l - 1).max(11);
        let (evals, evecs) = diffusion_map_from_kernel(
            &mut ll_kernel,
            n_eigs,
            seed,
            Some(self.params.lanczos_params),
        )?;

        let landmark_multiscale = determine_multiscale_space(&evals, &evecs, Some(10));
        let n_components = landmark_multiscale.first().map_or(0, Vec::len);
        let used_lambdas: Vec<f32> = (1..=n_components).map(|i| evals[i] as f32).collect();

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
    fn get_greedy_centres(&self, n_centres: usize) -> Result<Vec<usize>, BixverseErrors> {
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
                        let idx_usize = idx as usize;
                        row_i[kernel.indices[idx_usize] as usize] = kernel.data[idx_usize];
                    }
                    let k2_col_i = csr_matvec(kernel, &row_i)?;
                    Ok((i, k2_col_i))
                })
                .collect::<Result<Vec<(usize, Vec<f32>)>, BixverseErrors>>()?;
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
            let mut delta = self.k_squared_matvec(&e_p)?;

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
            let k_omega_new = self.k_squared_matvec(&omega_new)?;

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

        Ok(centres)
    }

    /// Initialise A and B matrices
    ///
    /// Creates:
    ///
    /// - B matrix (n × k): one-hot encoding of archetype cells
    /// - A matrix (k × n): sparse random assignments, column-L1-normalised
    ///
    /// Each cell is randomly assigned to `min(10, k)` archetypes with uniform
    /// random weights, then column-normalised so each cell's weights sum to 1.
    /// A is then refined by one full Frank-Wolfe update pass against the fixed
    /// B for a better starting point.
    ///
    /// The L1 column normalisation matches the Python reference; the fixed
    /// count is a deliberate deviation from it, since the reference scales the
    /// non-zeros per column with `k` at ⌈0.25 k⌉.
    ///
    /// ### Params
    ///
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    /// * `seed` - Random seed for A matrix initialisation
    /// * `backend` - Frank-Wolfe back end running the refinement pass
    fn initialise_matrices(
        &mut self,
        verbose: usize,
        seed: u64,
        backend: &mut impl FwArgminB,
    ) -> Result<(), BixverseErrors> {
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

        let b = coo_to_csr(&b_rows.index_cast(), &b_cols.index_cast(), &b_vals, (n, k));

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

        let mut a = coo_to_csr(&a_rows.index_cast(), &a_cols.index_cast(), &a_vals, (k, n));
        normalise_csr_columns_l1(&mut a);

        let k2_b = self.k_squared_matmul(&b)?;
        a = self.update_a_mat(&b, &k2_b, &a, verbose, backend)?;

        self.a = Some(a);
        self.b = Some(b);
        self.k2_b = Some(k2_b);

        Ok(())
    }

    /// Whether a cached `K² B` still matches its `B`.
    ///
    /// Only ever called from `debug_assert!`: it recomputes the product from
    /// scratch, which is exactly the work the cache exists to avoid. Every
    /// consumer of [SEACells::k2_b] trusts the invariant blindly, so an edit
    /// that writes `b` without writing `k2_b` would otherwise produce a
    /// silently wrong gradient and RSS.
    ///
    /// ### Params
    ///
    /// * `b` - The archetype matrix
    /// * `k2_b` - The cached product to check against it
    ///
    /// ### Returns
    ///
    /// `true` when the two agree to a scale-relative tolerance.
    ///
    /// Not `cfg(debug_assertions)`-gated: `debug_assert!` still compiles its
    /// argument in release, it just never evaluates it.
    fn k2_b_matches(
        &self,
        b: &CompressedSparseData2<f32>,
        k2_b: &CompressedSparseData2<f32>,
    ) -> bool {
        let Ok(reference) = self.k_squared_matmul(b) else {
            return false;
        };
        if reference.shape != k2_b.shape {
            return false;
        }

        let (nrow, ncol) = reference.shape;
        let mut dense = vec![0.0f32; nrow * ncol];
        for row in 0..nrow {
            for idx in reference.indptr[row] as usize..reference.indptr[row + 1] as usize {
                dense[row * ncol + reference.indices[idx] as usize] += reference.data[idx];
            }
        }
        let scale = dense.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));

        for row in 0..nrow {
            for idx in k2_b.indptr[row] as usize..k2_b.indptr[row + 1] as usize {
                dense[row * ncol + k2_b.indices[idx] as usize] -= k2_b.data[idx];
            }
        }
        let worst = dense.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));

        scale == 0.0 || worst / scale < 1e-3
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
    /// `K² B` is supplied by the caller rather than recomputed; see
    /// [SEACells::k2_b] for the invariant that makes that safe.
    ///
    /// For each cell (column), sets weight to 1 for the archetype with
    /// minimum gradient, then takes a convex step A ← (1 - γ) A + γ E
    /// with γ = 2/(t + 2).
    ///
    /// ### Params
    ///
    /// * `b` - Current archetype matrix
    /// * `k2_b` - `K² b`, matching `b`
    /// * `a_prev` - Previous assignment matrix
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    /// * `backend` - Frank-Wolfe back end running the column solve
    ///
    /// ### Returns
    ///
    /// Updated assignment matrix
    fn update_a_mat(
        &self,
        b: &CompressedSparseData2<f32>,
        k2_b: &CompressedSparseData2<f32>,
        a_prev: &CompressedSparseData2<f32>,
        verbose: usize,
        backend: &mut impl FwArgminB,
    ) -> Result<CompressedSparseData2<f32>, BixverseErrors> {
        let verbosity = parse_verbosity_level(verbose);

        let t2 = k2_b.transpose_and_convert();
        let t1 = csr_matmul_csr(&t2, b)?;
        drop(t2);
        let a_prev_t = a_prev.transpose_and_convert();

        let n = a_prev.shape.1;
        let k = a_prev.shape.0;

        let columns = backend.columns_a(
            &t1,
            &a_prev_t,
            k2_b,
            k,
            n,
            self.params.max_fw_iters,
            self.params.pruning.then_some(self.params.pruning_threshold),
        )?;

        let a = fw_atoms_to_csr(&columns, (k, n));

        if verbosity.detailed_verbosity() {
            println!(
                "  A matrix Frank-Wolfe: {} iterations over {} cells",
                self.params.max_fw_iters,
                n.separate_with_underscores()
            );
        }

        Ok(a)
    }

    /// Iteration-major A update, kept as the reference for the parity test
    ///
    /// This is the formulation `update_a_mat` used before the cell-major
    /// rewrite. It rebuilds the gradient from the sparse `A` every iteration
    /// and rebuilds `A` through an `E` matrix, a sort and a sparse add.
    /// Retained so the new path can be checked against it rather than against
    /// itself.
    ///
    /// ### Params
    ///
    /// * `b` - Current archetype matrix
    /// * `a_prev` - Previous assignment matrix
    ///
    /// ### Returns
    ///
    /// Updated assignment matrix
    #[cfg(test)]
    fn update_a_mat_iteration_major(
        &self,
        b: &CompressedSparseData2<f32>,
        a_prev: &CompressedSparseData2<f32>,
    ) -> Result<CompressedSparseData2<f32>, BixverseErrors> {
        let k2_b = self.k_squared_matmul(b)?;
        let t2 = k2_b.transpose_and_convert();
        let t1 = csr_matmul_csr(&t2, b)?;
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
            let e =
                coo_to_csr_presorted(&e_rows.index_cast(), &e_cols.index_cast(), &e_vals, (k, n));

            let step_size = 2.0 / (t as f32 + 2.0);
            let retain = 1.0 - step_size;
            for val in &mut a.data {
                *val *= retain;
            }
            let e_scaled = sparse_scalar_multiply_csr(&e, step_size);
            a = sparse_add_csr(&a, &e_scaled)?;

            if self.params.pruning {
                prune_and_renormalise(&mut a, self.params.pruning_threshold);
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
    /// `K² B` is carried through the inner loop rather than recomputed. Each
    /// Frank-Wolfe step changes `B` by a rank-k update, so the matching change
    /// to `K² B` is the same scaling plus one weighted column of `K²` per
    /// archetype, assembled by [k_squared_column_combination]. Pruning folds
    /// into the same delta. The update is exact; a full `K @ (K @ B)` recompute
    /// still runs every [K2B_REFRESH_EVERY] iterations to bound fp drift and
    /// the sparsity pattern.
    ///
    /// Stops early once the relative Frank-Wolfe duality gap falls below
    /// `FW_REL_TOL`, but never before [MIN_FW_ITERS] iterations have run.
    ///
    /// ### Params
    ///
    /// * `a` - Current assignment matrix
    /// * `b_prev` - Previous archetype matrix
    /// * `k2_b_prev` - `K² b_prev`, matching `b_prev`
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    /// * `backend` - Frank-Wolfe back end running the per-archetype argmin
    ///
    /// ### Returns
    ///
    /// The updated archetype matrix and its `K² B`, maintained together so the
    /// invariant on [SEACells::k2_b] survives the call.
    fn update_b_mat(
        &self,
        a: &CompressedSparseData2<f32>,
        b_prev: &CompressedSparseData2<f32>,
        k2_b_prev: &CompressedSparseData2<f32>,
        verbose: usize,
        backend: &mut impl FwArgminB,
    ) -> Result<(CompressedSparseData2<f32>, CompressedSparseData2<f32>), BixverseErrors> {
        let verbosity = parse_verbosity_level(verbose);

        const FW_REL_TOL: f32 = 1e-3;

        let kernel = self
            .kernel_mat
            .as_ref()
            .ok_or(BixverseErrors::SEACellsKernelMatrixMissing)?;

        let a_t = a.transpose_and_convert();
        let t1 = csr_matmul_csr(a, &a_t)?;
        let t2 = self.k_squared_matmul(&a_t)?;
        backend.begin(&t1, &t2)?;
        drop(t2);

        let mut b = b_prev.clone();
        let mut k2_b = k2_b_prev.clone();
        let n = b.shape.0;
        let k = b.shape.1;
        let mut initial_gap: f32 = 0.0;

        for t in 0..self.params.max_fw_iters {
            let (argmins, fw_gap) = backend.argmins(&k2_b, &b)?;
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
            let e =
                coo_to_csr_presorted(&e_rows.index_cast(), &e_cols.index_cast(), &e_vals, (n, k));

            let step_size = 2.0 / (t as f32 + 2.0);
            let retain = 1.0 - step_size;
            for val in &mut b.data {
                *val *= retain;
            }
            let e_scaled = sparse_scalar_multiply_csr(&e, step_size);
            b = sparse_add_csr(&b, &e_scaled)?;

            // `E` is one unit entry per column, so `K² E` is one column of `K²`
            // per archetype and the whole update is rank-k. Pruning folds into
            // the same delta: dropping `B[j, c] = v` contributes `-v K²[:, j]`.
            let mut contributions: Vec<Vec<(u32, f32)>> = vec![Vec::new(); k];
            for (col, &row) in argmins.iter().enumerate() {
                contributions[col].push((row as u32, step_size));
            }

            let renorm = if self.params.pruning {
                let (dropped, factors) =
                    prune_and_renormalise_tracked(&mut b, self.params.pruning_threshold);
                for (col, row, value) in dropped {
                    contributions[col as usize].push((row, -value));
                }
                Some(factors)
            } else {
                None
            };

            for val in &mut k2_b.data {
                *val *= retain;
            }
            let delta = k_squared_column_combination(kernel, &contributions, n);
            k2_b = sparse_add_csr(&k2_b, &transpose_sparse(&delta))?;
            if let Some(factors) = renorm {
                for (idx, &col) in k2_b.indices.iter().enumerate() {
                    k2_b.data[idx] *= factors[col as usize];
                }
            }

            // The incremental form is exact but only ever grows the sparsity
            // pattern, since entries decay without reaching zero.
            if (t + 1) % K2B_REFRESH_EVERY == 0 {
                k2_b = self.k_squared_matmul(&b)?;
            }

            if verbosity.detailed_verbosity() && (t + 1) % 10 == 0 {
                println!(
                    "  B matrix Frank-Wolfe iteration: {} / {}, nnz(B) = {}, nnz(K^2 B) = {}",
                    t + 1,
                    self.params.max_fw_iters,
                    b.get_nnz().separate_with_underscores(),
                    k2_b.get_nnz().separate_with_underscores()
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

        Ok((b, k2_b))
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
    /// * `k2_b` - `K² b`, matching `b`
    ///
    /// ### Returns
    ///
    /// RSS value (lower is better fit)
    fn compute_rss(
        &self,
        a: &CompressedSparseData2<f32>,
        b: &CompressedSparseData2<f32>,
        k2_b: &CompressedSparseData2<f32>,
    ) -> Result<f32, BixverseErrors> {
        self.compute_rss_trace(a, b, k2_b)
    }

    /// RSS by materialising the reconstruction
    ///
    /// Forms the n × n reconstruction K B A directly and returns the Frobenius
    /// norm of (K - K B A). Superseded by [SEACells::compute_rss_trace], which
    /// is faster at every size measured and agrees to well within the
    /// convergence threshold. Retained as the reference the trace identity is
    /// checked against.
    ///
    /// ### Params
    ///
    /// * `a` - The A matrix
    /// * `b` - The B matrix
    ///
    /// ### Returns
    ///
    /// The residual sum of squares (RSS)
    #[cfg(test)]
    fn compute_rss_simple(
        &self,
        a: &CompressedSparseData2<f32>,
        b: &CompressedSparseData2<f32>,
    ) -> Result<f32, BixverseErrors> {
        let k_mat = self.kernel_mat.as_ref().unwrap();
        let k_b = csr_matmul_csr(k_mat, b)?;
        let reconstruction = csr_matmul_csr(&k_b, a)?;
        let diff = sparse_subtract_csr(k_mat, &reconstruction)?;

        Ok(frobenius_norm(&diff))
    }

    /// Memory-efficient RSS computation for large datasets (uses trace trick)
    ///
    /// Expands the squared Frobenius norm via the trace identity:
    ///
    /// ```||K - K B A||_F² = ||K||_F² - 2 tr(K² B A) + tr(A Aᵀ Bᵀ K² B)```
    ///
    /// Cyclic trace reordering keeps every intermediate at worst (n × k) or
    /// (k × k); the n × n reconstruction is never formed. `K² B` comes from the
    /// caller rather than being recomputed, see [SEACells::k2_b].
    ///
    /// The final `.sqrt()` converts back to the Frobenius norm to match
    /// `compute_rss_simple`.
    ///
    /// ### Params
    ///
    /// * `a` - The A matrix
    /// * `b` - The B matrix
    /// * `k2_b` - `K² b`, matching `b`
    ///
    /// ### Returns
    ///
    /// The residual sum of squares (RSS)
    fn compute_rss_trace(
        &self,
        a: &CompressedSparseData2<f32>,
        b: &CompressedSparseData2<f32>,
        k2_b: &CompressedSparseData2<f32>,
    ) -> Result<f32, BixverseErrors> {
        // Term 1: ||K||_F^2, cached by construct_kernel_mat
        let k_frob_sq = self
            .k_frobenius_norm_sq
            .ok_or(BixverseErrors::SEACellsKernelMatrixMissing)?;

        // Term 2: -2 * trace(K^2 @ B @ A)
        // Reorder via cyclic property: trace(A @ K^2 @ B)  [k × k]
        let a_k2b = csr_matmul_csr(a, k2_b)?;
        let trace_term = matrix_trace(&a_k2b);

        // Term 3: trace(A^T @ B^T @ K^2 @ B @ A)
        // Reorder via cyclic property: trace(A @ A^T @ B^T @ K^2 @ B)
        let a_t = a.transpose_and_convert();
        let a_at = csr_matmul_csr(a, &a_t)?; // [k × k]

        let b_t = b.transpose_and_convert();
        let bt_k2b = csr_matmul_csr(&b_t, k2_b)?; // [k × k]

        let result = csr_matmul_csr(&a_at, &bt_k2b)?; // [k × k]
        let reconstruction_frob_sq = matrix_trace(&result);

        let residual_sq = k_frob_sq - 2.0 * trace_term + reconstruction_frob_sq;

        Ok(residual_sq.max(0.0).sqrt() as f32)
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
                let idx_usize = idx as usize;

                if a_csc.data[idx_usize] > max_val {
                    max_val = a_csc.data[idx_usize];
                    max_arch = a_csc.indices[idx_usize] as usize;
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
    /// Vector of cell indices selected as archetypes, or
    /// `SEACellsArchetypesMissing` if they have not been initialised yet
    pub fn get_archetypes(&self) -> Result<Vec<usize>, BixverseErrors> {
        if self.archetypes.is_none() {
            return Err(BixverseErrors::SEACellsArchetypesMissing);
        }

        Ok(self.archetypes.as_ref().unwrap().clone())
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Deterministic symmetric row-major `k × k` `t1` with a strong diagonal.
    fn dense_t1(k: usize) -> Vec<f32> {
        let mut t1 = vec![0.0f32; k * k];
        for i in 0..k {
            for j in i..k {
                let v = ((i * 7 + j * 13) % 11) as f32 * 0.1 + if i == j { 2.0 } else { 0.0 };
                t1[i * k + j] = v;
                t1[j * k + i] = v;
            }
        }
        t1
    }

    /// Reference `t1 · x` computed from scratch off the atom list.
    fn reference_product(t1: &[f32], k: usize, atoms: &FwAtoms) -> Vec<f32> {
        let (indices, weights) = atoms.atoms();
        let mut out = vec![0.0f32; k];
        for (&r, &v) in indices.iter().zip(weights.iter()) {
            for i in 0..k {
                out[i] += v * t1[r as usize * k + i];
            }
        }
        out
    }

    /// Apply the convex step to both the atoms and the incremental state.
    fn step_both(atoms: &mut FwAtoms, w: &mut [f32], t1: &[f32], k: usize, gamma: f32, amin: u32) {
        atoms.step(gamma, amin);
        let retain = 1.0 - gamma;
        let row = &t1[amin as usize * k..(amin as usize + 1) * k];
        for i in 0..k {
            w[i] = w[i] * retain + gamma * row[i];
        }
    }

    /// Apply prune and renormalise to both the atoms and the incremental state.
    fn prune_both(atoms: &mut FwAtoms, w: &mut [f32], t1: &[f32], k: usize, threshold: f32) {
        let outcome = atoms.prune(threshold);
        for (r, v) in outcome.dropped {
            let row = &t1[r as usize * k..(r as usize + 1) * k];
            for i in 0..k {
                w[i] -= v * row[i];
            }
        }
        if outcome.renorm != 1.0 {
            for value in w.iter_mut() {
                *value *= outcome.renorm;
            }
        }
    }

    /// The incrementally maintained gradient state must track a from-scratch
    /// recomputation through steps, prunes and renormalisations. This is the
    /// invariant the whole reformulation rests on.
    #[test]
    fn test_fw_atoms_gradient_tracks_reference_without_pruning() {
        let k = 12;
        let t1 = dense_t1(k);
        let mut atoms = FwAtoms::with_capacity(50);
        let mut w = vec![0.0f32; k];

        for t in 0..50u32 {
            let gamma = 2.0 / (t as f32 + 2.0);
            let amin = (t * 5 + 3) % k as u32;
            step_both(&mut atoms, &mut w, &t1, k, gamma, amin);

            let reference = reference_product(&t1, k, &atoms);
            for i in 0..k {
                assert_relative_eq!(w[i], reference[i], max_relative = 1e-4, epsilon = 1e-6);
            }
        }

        let (_, weights) = atoms.atoms();
        let mass: f32 = weights.iter().sum();
        assert_relative_eq!(mass, 1.0, max_relative = 1e-5);
    }

    /// Same invariant with a threshold that fires on most iterations, which is the
    /// case the rank-1 removal exists for.
    #[test]
    fn test_fw_atoms_gradient_tracks_reference_with_pruning() {
        let k = 12;
        let t1 = dense_t1(k);
        let mut atoms = FwAtoms::with_capacity(50);
        let mut w = vec![0.0f32; k];

        // 5e-2 is well above the smallest weight the schedule produces, so atoms
        // that stop being chosen get dropped rather than merely decaying.
        let threshold = 5e-2f32;

        for t in 0..50u32 {
            let gamma = 2.0 / (t as f32 + 2.0);
            let amin = (t * 5 + 3) % k as u32;
            step_both(&mut atoms, &mut w, &t1, k, gamma, amin);
            prune_both(&mut atoms, &mut w, &t1, k, threshold);

            let reference = reference_product(&t1, k, &atoms);
            for i in 0..k {
                assert_relative_eq!(w[i], reference[i], max_relative = 1e-3, epsilon = 1e-5);
            }
        }

        assert!(
            !atoms.is_empty(),
            "pruning should not empty the column at this threshold"
        );
        let (_, weights) = atoms.atoms();
        let mass: f32 = weights.iter().sum();
        assert_relative_eq!(mass, 1.0, max_relative = 1e-4);
    }

    /// An atom dropped and later re-chosen must come back as a fresh atom, not as
    /// a stale weight, and the gradient state must follow.
    #[test]
    fn test_fw_atoms_dropped_atom_can_return() {
        let k = 8;
        let t1 = dense_t1(k);
        let mut atoms = FwAtoms::with_capacity(16);
        let mut w = vec![0.0f32; k];

        step_both(&mut atoms, &mut w, &t1, k, 1.0, 3);
        step_both(&mut atoms, &mut w, &t1, k, 0.5, 5);
        // Drops atom 3 (weight 0.5 after the step) is not wanted here; pick a
        // threshold that leaves both, then one that removes only atom 3.
        prune_both(&mut atoms, &mut w, &t1, k, 0.1);
        assert_eq!(atoms.len(), 2);

        step_both(&mut atoms, &mut w, &t1, k, 0.9, 5);
        prune_both(&mut atoms, &mut w, &t1, k, 0.2);
        assert_eq!(atoms.len(), 1, "atom 3 should have been pruned");
        assert_eq!(atoms.atoms().0[0], 5);

        // Re-choose the dropped atom.
        step_both(&mut atoms, &mut w, &t1, k, 0.5, 3);
        assert_eq!(atoms.len(), 2);

        let reference = reference_product(&t1, k, &atoms);
        for i in 0..k {
            assert_relative_eq!(w[i], reference[i], max_relative = 1e-4, epsilon = 1e-6);
        }
    }

    /// A threshold above every weight empties the column. The renormalisation must
    /// then be skipped rather than dividing by zero, matching
    /// `normalise_csr_columns_l1`.
    #[test]
    fn test_fw_atoms_prune_everything_leaves_zero_state() {
        let k = 6;
        let t1 = dense_t1(k);
        let mut atoms = FwAtoms::with_capacity(8);
        let mut w = vec![0.0f32; k];

        step_both(&mut atoms, &mut w, &t1, k, 1.0, 2);
        prune_both(&mut atoms, &mut w, &t1, k, 2.0);

        assert!(atoms.is_empty());
        for value in &w {
            assert_relative_eq!(*value, 0.0, epsilon = 1e-6);
        }
    }

    /// Repeated argmins merge rather than duplicating, which is what
    /// `sparse_add_csr` does on the iteration-major path.
    #[test]
    fn test_fw_atoms_repeated_argmin_merges() {
        let mut atoms = FwAtoms::with_capacity(8);
        atoms.step(1.0, 4);
        atoms.step(0.5, 4);
        atoms.step(0.25, 4);

        assert_eq!(atoms.len(), 1);
        let (indices, weights) = atoms.atoms();
        assert_eq!(indices[0], 4);
        assert_relative_eq!(weights[0], 1.0, max_relative = 1e-6);
    }

    /// Banded symmetric `n × n` CSR kernel with unit diagonal, standing in for
    /// the adaptive RBF kernel over a kNN graph.
    fn banded_kernel(n: usize, bandwidth: usize) -> CompressedSparseData2<f32> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for i in 0..n {
            let lo = i.saturating_sub(bandwidth);
            let hi = (i + bandwidth + 1).min(n);
            for j in lo..hi {
                let d = (i as f32 - j as f32).abs();
                rows.push(i);
                cols.push(j);
                vals.push((-d * d / 4.0).exp());
            }
        }

        coo_to_csr(&rows.index_cast(), &cols.index_cast(), &vals, (n, n))
    }

    /// Densify a CSR matrix into `nrow * ncol` row-major values.
    fn densify(mat: &CompressedSparseData2<f32>) -> Vec<f32> {
        let (nrow, ncol) = mat.shape;
        let mut out = vec![0.0f32; nrow * ncol];
        for row in 0..nrow {
            for idx in mat.indptr[row] as usize..mat.indptr[row + 1] as usize {
                out[row * ncol + mat.indices[idx] as usize] += mat.data[idx];
            }
        }
        out
    }

    /// The `(B, A_0)` pair, shaped `(n × k, k × n)`, the way `initialise_matrices` does it.
    fn initial_matrices(
        n: usize,
        k: usize,
        seed: u64,
    ) -> (CompressedSparseData2<f32>, CompressedSparseData2<f32>) {
        let stride = n / k;
        let b_rows: Vec<usize> = (0..k).map(|c| c * stride).collect();
        let b_cols: Vec<usize> = (0..k).collect();
        let b_vals = vec![1.0f32; k];
        let b = coo_to_csr(&b_rows.index_cast(), &b_cols.index_cast(), &b_vals, (n, k));

        let mut rng = StdRng::seed_from_u64(seed);
        let per_cell = 10.min(k);
        let mut a_rows = Vec::new();
        let mut a_cols = Vec::new();
        let mut a_vals = Vec::new();
        for cell in 0..n {
            for _ in 0..per_cell {
                a_rows.push(rng.random_range(0..k));
                a_cols.push(cell);
                a_vals.push(rng.random::<f32>());
            }
        }
        let mut a = coo_to_csr(&a_rows.index_cast(), &a_cols.index_cast(), &a_vals, (k, n));
        normalise_csr_columns_l1(&mut a);

        (b, a)
    }

    /// The incrementally maintained `K² B` must equal the from-scratch product.
    ///
    /// This is the gate for the whole incremental scheme. `update_b_mat` never
    /// recomputes `K² B`, it rides the rank-k identity
    /// `K² B_{t+1} = (1-γ) K² B_t + γ K²E_t` and folds the pruning corrections
    /// into the same delta, so if the maintenance drifts every downstream
    /// gradient and the RSS drift with it and nothing else would notice.
    ///
    /// Swept over the [K2B_REFRESH_EVERY] boundary so the assertion is about the
    /// incremental path rather than a full recompute that happened to land on
    /// the last iteration, and over both pruning settings because the dropped
    /// entries are the part most likely to be mirrored wrongly.
    #[test]
    fn test_k2b_maintenance_matches_from_scratch() {
        let n = 600usize;
        let k = 12usize;

        // The refresh firing at all is guaranteed by the `K2B_REFRESH_EVERY <
        // MIN_FW_ITERS` compile-time assertion next to those constants: the
        // convergence break cannot trigger before `MIN_FW_ITERS`, so the loop
        // always reaches the refresh. The branch itself is not observable here.

        for (max_fw_iters, pruning, threshold) in [
            // Below the refresh interval: purely incremental, nothing recomputed.
            (K2B_REFRESH_EVERY - 1, false, 0.0f32),
            (K2B_REFRESH_EVERY - 1, true, 1e-7),
            // Several incremental steps *after* a refresh, so the tail being
            // gated is incremental rather than a fresh product.
            (K2B_REFRESH_EVERY + 5, false, 0.0),
            (K2B_REFRESH_EVERY + 5, true, 1e-7),
            // Fires hard enough that entries are actually dropped, which is the
            // path the correction terms exist for.
            (K2B_REFRESH_EVERY + 4, true, 5e-2),
        ] {
            let params = SEACellsParams {
                lanczos_params: LanczosParams::default(),
                n_sea_cells: k,
                max_fw_iters,
                convergence_epsilon: 1e-3,
                max_iter: 2,
                min_iter: 2,
                greedy_threshold: 0,
                graph_building: "union".to_string(),
                pruning,
                pruning_threshold: threshold,
                n_landmarks: None,
                knn_params: KnnParams::new(),
            };

            let mut model = SEACells::new(n, &params);
            let kernel = banded_kernel(n, 3);
            model.k_frobenius_norm_sq = Some(frobenius_norm_sq_f64(&kernel));
            model.kernel_mat = Some(kernel);

            let (b, a_prev) = initial_matrices(n, k, 11);
            let k2_b = model.k_squared_matmul(&b).expect("K^2 B failed");
            let a = model
                .update_a_mat(&b, &k2_b, &a_prev, 0, &mut CpuFwArgminB::default())
                .expect("A update failed");

            let (b_new, k2_b_new) = model
                .update_b_mat(&a, &b, &k2_b, 0, &mut CpuFwArgminB::default())
                .expect("B update failed");

            let reference = model.k_squared_matmul(&b_new).expect("K^2 B failed");

            assert_eq!(k2_b_new.shape, reference.shape);

            let maintained = densify(&k2_b_new);
            let expected = densify(&reference);

            let scale = expected.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            assert!(scale > 0.0, "reference K^2 B is all zeros, test is vacuous");

            // Per entry, not against the global maximum. The entries span
            // several orders of magnitude, so a tolerance scaled to the largest
            // one lets anything small be arbitrarily wrong. The absolute floor
            // is what keeps near-zero entries from failing on pure rounding.
            let floor = 1e-6 * scale;
            let mut worst_rel = 0.0f32;
            let mut worst_at = (0usize, 0usize);
            for row in 0..expected.len() / k2_b_new.shape.1 {
                for col in 0..k2_b_new.shape.1 {
                    let i = row * k2_b_new.shape.1 + col;
                    let diff = (maintained[i] - expected[i]).abs();
                    if diff <= floor {
                        continue;
                    }
                    let rel = diff / expected[i].abs().max(floor);
                    if rel > worst_rel {
                        worst_rel = rel;
                        worst_at = (row, col);
                    }
                }
            }

            assert!(
                worst_rel < 1e-3,
                "K^2 B drifted from the from-scratch product: worst relative {:.3e} at \
                 ({}, {}) (max_fw_iters {}, pruning {:?}, threshold {})",
                worst_rel,
                worst_at.0,
                worst_at.1,
                max_fw_iters,
                pruning,
                threshold
            );
        }
    }

    /// The cell-major A update must reproduce the iteration-major one it replaces,
    /// at every pruning setting: pruning off, a threshold that never fires, and
    /// one that fires hard.
    ///
    /// Exact equality is deliberately *not* the invariant. The two paths compute
    /// the same gradient by different associations, so they differ in the last
    /// bits and flip near-ties; the maintenance itself is pinned exactly by
    /// `test_fw_atoms_gradient_tracks_reference_*`. Above `2/(T(T+1))`, the
    /// smallest weight the schedule produces, pruning also removes live mass and
    /// the renormalisation feeds it back, so a flipped tie cascades into a
    /// different but equally good vertex. Sparsity and objective are what hold.
    #[test]
    fn test_update_a_mat_matches_iteration_major() {
        let n = 300usize;
        let k = 12usize;

        for (pruning, threshold) in [(false, 0.0f32), (true, 1e-7), (true, 5e-2)] {
            let params = SEACellsParams {
                lanczos_params: LanczosParams::default(),
                n_sea_cells: k,
                max_fw_iters: 50,
                convergence_epsilon: 1e-3,
                max_iter: 3,
                min_iter: 1,
                greedy_threshold: 0,
                graph_building: "union".to_string(),
                pruning,
                pruning_threshold: threshold,
                n_landmarks: None,
                knn_params: KnnParams::new(),
            };

            let mut model = SEACells::new(n, &params);
            let kernel = banded_kernel(n, 4);
            model.k_frobenius_norm_sq = Some(frobenius_norm_sq_f64(&kernel));
            model.kernel_mat = Some(kernel);

            let (b, a_prev) = initial_matrices(n, k, 11);

            let k2_b = model.k_squared_matmul(&b).expect("K^2 B failed");
            let a_new = model
                .update_a_mat(&b, &k2_b, &a_prev, 0, &mut CpuFwArgminB::default())
                .expect("cell-major update failed");
            let a_ref = model
                .update_a_mat_iteration_major(&b, &a_prev)
                .expect("iteration-major update failed");

            assert_eq!(a_new.shape, a_ref.shape);

            let dense_new = densify(&a_new);
            let dense_ref = densify(&a_ref);

            // Disagreement must stay at the tie-break level, not the systematic
            // level. Anything above a couple of percent of cells means the
            // gradient maintenance is wrong rather than merely reassociated.
            let differing = (0..n)
                .filter(|&cell| {
                    (0..k).any(|r| (dense_new[r * n + cell] - dense_ref[r * n + cell]).abs() > 1e-4)
                })
                .count();
            // Below the smallest weight the schedule produces, pruning removes
            // only structural zeros and cannot feed back into the trajectory, so
            // the two paths must track each other cell by cell. Above it,
            // renormalisation redistributes mass and amplifies last-bit
            // differences into different-but-equally-good vertices, which the
            // objective check below is what pins.
            let t_iters = params.max_fw_iters as f32;
            let min_fw_weight = 2.0 / (t_iters * (t_iters + 1.0));
            if !pruning || threshold < min_fw_weight {
                assert!(
                    differing * 50 <= n,
                    "{} / {} cells differ (pruning {:?}, threshold {}) below the \
                     cascade threshold {:.2e}, well past a tie-break rate",
                    differing,
                    n,
                    pruning,
                    threshold,
                    min_fw_weight
                );
            }

            // Every column must still be a convex combination.
            for cell in 0..n {
                let mass: f32 = (0..k).map(|r| dense_new[r * n + cell]).sum();
                assert_relative_eq!(mass, 1.0, max_relative = 1e-4);
            }

            // The objective is what the argmin is a means to. A flipped near-tie
            // must not move it.
            let rss_new = model.compute_rss(&a_new, &b, &k2_b).expect("RSS failed");
            let rss_ref = model.compute_rss(&a_ref, &b, &k2_b).expect("RSS failed");
            assert_relative_eq!(rss_new, rss_ref, max_relative = 1e-4);

            // Memory regression guard: the rewrite must not densify. The slack
            // covers the one extra atom a flipped tie can introduce per cell.
            assert!(
                a_new.get_nnz() <= a_ref.get_nnz() + differing,
                "nnz grew beyond the tie-break slack: {} vs {} + {} (pruning {:?}, threshold {})",
                a_new.get_nnz(),
                a_ref.get_nnz(),
                differing,
                pruning,
                threshold
            );
        }
    }

    /// With pruning off, the atom weights collapse to `2(t+1) / (T(T+1))` when
    /// every argmin is distinct. This is the closed form the GPU fast arm uses.
    #[test]
    fn test_fw_atoms_closed_form_weights() {
        let n_iters = 20usize;
        let mut atoms = FwAtoms::with_capacity(n_iters);

        for t in 0..n_iters {
            let gamma = 2.0 / (t as f32 + 2.0);
            atoms.step(gamma, t as u32);
        }

        let denom = (n_iters * (n_iters + 1)) as f32;
        let (indices, weights) = atoms.atoms();
        assert_eq!(indices.len(), n_iters);
        for (t, &weight) in weights.iter().enumerate() {
            let expected = 2.0 * (t as f32 + 1.0) / denom;
            assert_relative_eq!(weight, expected, max_relative = 1e-5);
        }
    }

    /// The regime the trace identity is dangerous in: `k = n`, so `B` is the
    /// identity and the residual is a small fraction of `||K||_F`. The three terms
    /// then cancel almost completely and the relative error is worst.
    ///
    /// The bound asserted here is loose on purpose. What has to hold is that the
    /// result stays finite and non-negative rather than turning into a NaN that
    /// would make the convergence test silently false forever.
    #[test]
    fn test_rss_trace_survives_near_total_cancellation() {
        let n = 200usize;
        let k = 200usize;
        let params = SEACellsParams {
            lanczos_params: LanczosParams::default(),
            n_sea_cells: k,
            max_fw_iters: 50,
            convergence_epsilon: 1e-3,
            max_iter: 3,
            min_iter: 1,
            greedy_threshold: 0,
            graph_building: "union".to_string(),
            pruning: false,
            pruning_threshold: 0.0,
            n_landmarks: None,
            knn_params: KnnParams::new(),
        };

        for bandwidth in [0usize, 2] {
            let mut model = SEACells::new(n, &params);
            let kernel = banded_kernel(n, bandwidth);
            model.k_frobenius_norm_sq = Some(frobenius_norm_sq_f64(&kernel));
            model.kernel_mat = Some(kernel);

            let (b, a_prev) = initial_matrices(n, k, 11);
            let k2_b = model.k_squared_matmul(&b).expect("K^2 B failed");
            let a = model
                .update_a_mat(&b, &k2_b, &a_prev, 0, &mut CpuFwArgminB::default())
                .expect("A update failed");

            let simple = model.compute_rss_simple(&a, &b).expect("simple failed");
            let trace = model
                .compute_rss_trace(&a, &b, &k2_b)
                .expect("trace failed");

            let residual_fraction = simple / model.k_frobenius_norm_sq.unwrap().sqrt() as f32;
            println!(
                "bandwidth {} | simple {:.6} trace {:.6} | residual {:.3}% of |K|_F | rel {:.2e}",
                bandwidth,
                simple,
                trace,
                100.0 * residual_fraction,
                ((simple - trace) / simple).abs()
            );

            assert!(
                trace.is_finite() && trace >= 0.0,
                "trace RSS is not a usable number at bandwidth {}: {}",
                bandwidth,
                trace
            );
            assert!(
                ((simple - trace) / simple).abs() < 5e-2,
                "trace RSS lost its accuracy at bandwidth {}: {} vs {}",
                bandwidth,
                trace,
                simple
            );
        }
    }

    /// Compare the two RSS paths on accuracy and cost.
    ///
    /// The trace path expands `||K - KBA||_F^2` into three terms that
    /// individually dwarf the result, so cancellation is the risk that would
    /// justify keeping the materialising path. It does not materialise: the two
    /// agree well inside the convergence threshold at every size tested.
    ///
    /// The kernel is dense, so the materialising path costs `O(n^2)` and the two
    /// larger sizes were five seconds of the crate's test time on their own.
    /// They stay behind `large-test`; 2000 cells run everywhere and
    /// assert the same bound. `test_rss_trace_survives_near_total_cancellation`
    /// covers the regime where the cancellation is worst, which is `k = n`
    /// rather than large `n`.
    #[test]
    fn test_rss_paths_agree() {
        #[cfg(not(feature = "large-test"))]
        let sizes: &[usize] = &[2000];
        #[cfg(feature = "large-test")]
        let sizes: &[usize] = &[2000, 8000, 20000];

        for &n in sizes {
            let k = (n / 75).max(8);
            let params = SEACellsParams {
                lanczos_params: LanczosParams::default(),
                n_sea_cells: k,
                max_fw_iters: 50,
                convergence_epsilon: 1e-3,
                max_iter: 3,
                min_iter: 1,
                greedy_threshold: 0,
                graph_building: "union".to_string(),
                pruning: false,
                pruning_threshold: 0.0,
                n_landmarks: None,
                knn_params: KnnParams::new(),
            };
            let mut model = SEACells::new(n, &params);
            let kernel = banded_kernel(n, 6);
            model.k_frobenius_norm_sq = Some(frobenius_norm_sq_f64(&kernel));
            model.kernel_mat = Some(kernel);

            let (b, a_prev) = initial_matrices(n, k, 11);
            let k2_b = model.k_squared_matmul(&b).expect("K^2 B failed");
            let a = model
                .update_a_mat(&b, &k2_b, &a_prev, 0, &mut CpuFwArgminB::default())
                .expect("A update failed");

            let simple_start = Instant::now();
            let simple = model.compute_rss_simple(&a, &b).expect("simple failed");
            let simple_time = simple_start.elapsed();

            let trace_start = Instant::now();
            let trace = model
                .compute_rss_trace(&a, &b, &k2_b)
                .expect("trace failed");
            let trace_time = trace_start.elapsed();

            let rel = ((simple - trace) / simple).abs();
            println!(
                "n = {:>6} k = {:>4} | simple {:>8.4} in {:>8.3}s | trace {:>8.4} in {:>8.3}s \
                 | speedup {:>7.1}x | rel diff {:.2e}",
                n,
                k,
                simple,
                simple_time.as_secs_f64(),
                trace,
                trace_time.as_secs_f64(),
                simple_time.as_secs_f64() / trace_time.as_secs_f64(),
                rel
            );

            // Tied to what the trace path has to protect: the convergence test
            // fires on `|RSS_{i-1} - RSS_i| < convergence_epsilon * RSS_0`, so
            // the RSS noise has to sit well under 1e-3. Measured 4e-6 to 8e-6.
            assert!(
                rel < 1e-4,
                "RSS paths disagree at n = {}: {} vs {} (rel {:.2e})",
                n,
                simple,
                trace,
                rel
            );
        }
    }

    /// A ring kNN graph, `k` neighbours per cell, as `(indices, distances)`.
    fn ring_knn(n: usize, k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let indices: Vec<Vec<usize>> = (0..n)
            .map(|i| (1..=k).map(|step| (i + step) % n).collect())
            .collect();
        let distances: Vec<Vec<f32>> = (0..n)
            .map(|_| (1..=k).map(|step| step as f32).collect())
            .collect();
        (indices, distances)
    }

    /// Regression: a graph narrower than the caller's `k` used to index past the row.
    #[test]
    fn test_diffusion_kernel_narrow_graph() {
        // A five-neighbour graph used to index `sorted[7]` when the caller's
        // params asked for k = 25. The bandwidth now comes from the row itself.
        let (indices, distances) = ring_knn(20, 5);
        let kernel = compute_diffusion_kernel(&indices, &distances, false)
            .expect("narrow kNN graph must not panic");
        assert_eq!(kernel.shape, (20, 20));
    }

    /// The bandwidth is taken from the row's own width, not from a fixed parameter.
    #[test]
    fn test_diffusion_kernel_bandwidth_scales_with_row_width() {
        // adaptive_k is (row width / 3).max(1), so a 3-wide row picks the
        // first sorted distance and a 9-wide row picks the third.
        let (idx_narrow, dist_narrow) = ring_knn(12, 3);
        let (idx_wide, dist_wide) = ring_knn(12, 9);

        let narrow = compute_diffusion_kernel(&idx_narrow, &dist_narrow, false).unwrap();
        let wide = compute_diffusion_kernel(&idx_wide, &dist_wide, false).unwrap();

        // exp(-1/1) for the narrow graph, exp(-1/3) for the wide one, on the
        // nearest neighbour of cell 0. Different bandwidths, same edge.
        let narrow_first = narrow.data[0];
        let wide_first = wide.data[0];
        assert!(
            wide_first > narrow_first,
            "wider graph should give a larger bandwidth and heavier weights: {narrow_first} vs {wide_first}"
        );
    }

    /// A cell with no neighbours has no bandwidth to take, so it must error not panic.
    #[test]
    fn test_diffusion_kernel_empty_row_errors() {
        let indices = vec![vec![1usize], vec![]];
        let distances = vec![vec![1.0f32], vec![]];
        assert!(compute_diffusion_kernel(&indices, &distances, false).is_err());
    }

    /// A NaN distance is caught at the kernel, where it can still name its cause.
    #[test]
    fn test_diffusion_kernel_nan_distance_errors_rather_than_panics() {
        let (indices, mut distances) = ring_knn(10, 6);
        distances[3][2] = f32::NAN;

        // total_cmp sorts NaN to the end rather than panicking the comparator,
        // so the bandwidth pass survives. The NaN then reaches the weight and
        // has to be caught here: `f32::min` / `f32::max` swallow it in the
        // min-max scaling, every NaN comparison in the boundary scan is false,
        // and the panic finally lands in the waypoint sampler with nothing left
        // pointing back at the kNN input.
        assert!(matches!(
            compute_diffusion_kernel(&indices, &distances, false),
            Err(BixverseErrors::DiffusionKernelNotFinite { n_cells: 10 })
        ));
    }

    /// A one-dimensional kNN fixture with cell 1 sitting exactly on cell 0,
    /// returning un-squared distances. Four supplied neighbours puts the
    /// bandwidth rank at `((4 + 1) / 3).max(1) - 1 = 0`, the nearest neighbour,
    /// which for both duplicates is a distance of zero. `mutual = false` drops
    /// cell 0 from cell 1's row, which is what an approximate backend returning
    /// an asymmetric graph looks like.
    fn duplicate_cell_knn(mutual: bool) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let n = 8usize;
        let coords: Vec<f32> = (0..n)
            .map(|i| if i == 1 { 0.0 } else { i as f32 })
            .collect();

        let mut indices = Vec::with_capacity(n);
        let mut distances = Vec::with_capacity(n);
        for i in 0..n {
            let mut order: Vec<usize> = (0..n)
                .filter(|&j| j != i && (mutual || i != 1 || j != 0))
                .collect();
            order.sort_by(|&a, &b| {
                (coords[a] - coords[i])
                    .abs()
                    .total_cmp(&(coords[b] - coords[i]).abs())
            });
            order.truncate(4);
            distances.push(
                order
                    .iter()
                    .map(|&j| (coords[j] - coords[i]).abs())
                    .collect::<Vec<f32>>(),
            );
            indices.push(order);
        }

        (indices, distances)
    }

    /// Regression: a zero bandwidth used to gut the duplicated cell's row silently.
    #[test]
    fn test_diffusion_kernel_keeps_the_row_of_a_duplicated_cell() {
        // A zero bandwidth turns every one of that cell's outgoing weights into
        // `exp(-d / 0)`, which is an exact zero for `d > 0` and NaN for the
        // duplicate itself. `coo_to_csr` drops the zeros and `sparse_add_csr`
        // drops the NaN, since `NaN.abs() > EPSILON` is false, so the row is
        // silently gutted rather than blowing up. That is the reference's
        // behaviour and it is what the smallest-positive fallback exists to
        // avoid.
        let (indices, distances) = duplicate_cell_knn(true);
        assert_eq!(distances[0][0], 0.0);

        let kernel = compute_diffusion_kernel(&indices, &distances, false).unwrap();

        assert!(kernel.data.iter().all(|v| v.is_finite()));

        // Cell 0 keeps every neighbour it was given, the duplicate included.
        let lo = kernel.indptr[0] as usize;
        let hi = kernel.indptr[1] as usize;
        assert!(
            kernel.indices[lo..hi].contains(&1),
            "the edge to the duplicate was dropped"
        );
        assert_eq!(hi - lo, 4, "cell 0 lost neighbours to a zero bandwidth");
        assert!(kernel.data[lo..hi].iter().all(|&v| v > 0.0));
    }

    /// Regression: an asymmetric duplicate pair leaked a NaN nothing downstream could see.
    #[test]
    fn test_diffusion_kernel_survives_a_one_sided_duplicate() {
        // The dangerous shape. With the duplicate pair stored both ways the
        // `0 -> 1` NaN meets the `1 -> 0` NaN in `sparse_add_csr`, sums to NaN,
        // and is dropped because `NaN.abs() > EPSILON` is false. Stored one way
        // only, it never meets a partner and reaches the output, where nothing
        // downstream can see it: `f32::min` / `f32::max` swallow it in the
        // min-max scaling, every NaN comparison in the boundary scan is false,
        // and the panic finally lands in `max_min_sampling` naming nothing.
        let (indices, distances) = duplicate_cell_knn(false);

        // Cell 0 gets a zero bandwidth, cell 1 does not, so only the 0 -> 1
        // direction is degenerate.
        assert_eq!(distances[0][0], 0.0);
        assert!(distances[1][0] > 0.0);

        let kernel = compute_diffusion_kernel(&indices, &distances, false).unwrap();

        assert!(kernel.data.iter().all(|v| v.is_finite()));
        let lo = kernel.indptr[0] as usize;
        let hi = kernel.indptr[1] as usize;
        assert!(kernel.indices[lo..hi].contains(&1));
    }

    /// The bandwidth rank counts the self edge, matching what the scanpy reference does.
    #[test]
    fn test_diffusion_kernel_bandwidth_rank_is_self_inclusive() {
        // The reference asks scanpy for `n_neighbors = 30`, gets 29 stored
        // distances and indexes rank `floor(30 / 3) - 1 = 9`. Feeding 29
        // neighbours here has to land on the same rank, not on 8.
        let n = 40usize;
        let mut indices = Vec::with_capacity(n);
        let mut distances = Vec::with_capacity(n);
        for i in 0..n {
            let mut order: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            order.sort_by_key(|&j| j.abs_diff(i));
            order.truncate(29);
            distances.push(
                order
                    .iter()
                    .map(|&j| j.abs_diff(i) as f32)
                    .collect::<Vec<f32>>(),
            );
            indices.push(order);
        }

        let kernel = compute_diffusion_kernel(&indices, &distances, false).unwrap();

        // Cell 20 sits in the middle, so its sorted distances run
        // 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, ... and rank 9 is 5 while rank 8 is 4.
        // The self edge is absent, so the nearest stored weight is exp(-1/sigma)
        // doubled by the symmetrisation.
        let lo = kernel.indptr[20] as usize;
        let hi = kernel.indptr[20 + 1] as usize;
        let nearest = kernel.data[lo..hi]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        assert_relative_eq!(nearest, 2.0 * (-1.0f32 / 5.0).exp(), epsilon = 1e-6);
    }
}
