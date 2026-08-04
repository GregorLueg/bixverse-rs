//! Metrics used for assessing meta cell separation and compactness and if they
//! sit in high, mid or low density regions of the manifold, represented by
//! diffusion maps. This metrics have been ported from Persad, et al., Nat.
//! Biotechnol., 2023.

use ann_search_rs::utils::dist::euclidean_distance_static;
use faer::{Mat, MatRef};
use rayon::prelude::*;

use crate::prelude::*;
use crate::single_cell::sc_processing::knn::{KnnParams, generate_knn_with_dist};

use crate::single_cell::mc_generation::seacells::{
    compute_diffusion_kernel, determine_multiscale_space, diffusion_map_from_kernel,
};

//////////////////////////
// Enums and structures //
//////////////////////////

/// Density region classification per the SEACells benchmarking convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DensityRegion {
    /// Low density region of the manifold
    Low,
    /// Mid density region of the manifold
    Mid,
    /// High density region of the manifold
    High,
}

/// Diffusion components together with per-cell density classification.
pub struct DiffusionDensity {
    /// Multiscale diffusion components, shape n × n_dcs.
    pub dcs: Mat<f32>,
    /// Distance to the k_density-th nearest neighbour in DC space.
    pub density_distances: Vec<f32>,
    /// Per-cell density bucket: lower quartile of distances → High,
    /// upper quartile → Low, middle → Mid.
    pub regions: Vec<DensityRegion>,
}

/////////////
// Helpers //
/////////////

/// Quartile-based density classification on raw distances.
///
/// Lower quartile distances → High (densest), upper quartile → Low (sparsest),
/// remainder → Mid. No metacell-level cap is applied here; the paper's 30%
/// cap is a downstream metacell aggregation step, not a cell-level concern.
///
/// ### Params
///
/// * `dist` - The distances
///
/// ### Returns
///
/// A Vec of [DensityRegion]
fn classify_density_regions(dist: &[f32]) -> Vec<DensityRegion> {
    let n = dist.len();
    let mut sorted: Vec<(usize, f32)> = dist.iter().copied().enumerate().collect();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let q1 = n / 4;
    let q3 = 3 * n / 4;

    let mut regions = vec![DensityRegion::Mid; n];
    for &(idx, _) in &sorted[..q1] {
        regions[idx] = DensityRegion::High;
    }
    for &(idx, _) in &sorted[q3..] {
        regions[idx] = DensityRegion::Low;
    }
    regions
}

//////////
// Main //
//////////

/// Compute top diffusion components and tag each cell as low / mid / high density.
///
/// Follows the SEACells benchmarking definition:
///
/// 1. Adaptive bandwidth diffusion kernel from input kNN graph.
/// 2. Eigendecompose, then apply Palantir multiscale scaling λ/(1-λ).
/// 3. kNN search on the DC embedding with `k_density` neighbours.
/// 4. Distance to the k_density-th neighbour is the density proxy.
/// 5. Lower quartile of distances → high density; upper quartile → low density.
///
/// ### Params
///
/// * `knn_indices` - Input kNN indices used to build the diffusion kernel
/// * `knn_distances` - Input kNN distances.
/// * `squared_dist` - If the distance is squared (for example Euclidean
///   squared).
/// * `n_dcs` - Number of diffusion components to retain (paper uses `10`)
/// * `k_density` - Neighbour rank for the density estimate (paper uses `150`)
/// * `knn_params` - Parameters for the DC-space kNN search
/// * `seed` - RNG seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The [DiffusionDensity] results.
#[allow(clippy::too_many_arguments)]
pub fn compute_diffusion_density(
    knn_indices: &[Vec<usize>],
    knn_distances: &[Vec<f32>],
    squared_dist: bool,
    n_dcs: usize,
    k_density: usize,
    knn_params: &KnnParams,
    seed: u64,
    verbose: usize,
) -> Result<DiffusionDensity, BixverseErrors> {
    let n = knn_indices.len();

    let verbosity = parse_verbosity_level(verbose);

    if verbosity.normal_verbosity() {
        println!("Building diffusion kernel...");
    }
    let mut kernel = compute_diffusion_kernel(knn_indices, knn_distances, squared_dist)?;

    if verbosity.normal_verbosity() {
        println!("Computing top {} diffusion components...", n_dcs);
    }
    let (evals, evecs) = diffusion_map_from_kernel(&mut kernel, n_dcs + 1, seed)?;
    let dcs = determine_multiscale_space(&evals, &evecs, Some(n_dcs + 1));
    let dim = dcs[0].len();

    if verbosity.normal_verbosity() {
        println!("Running kNN on DC embedding (k = {})...", k_density);
    }
    let dc_mat = Mat::<f32>::from_fn(n, dim, |i, j| dcs[i][j]);

    let mut params = knn_params.clone();
    params.k = k_density.min(n.saturating_sub(1)).max(1);
    params.ann_dist = "euclidean".to_string();

    let (_, dc_distances) = generate_knn_with_dist(
        dc_mat.as_ref(),
        &params,
        true,
        false,
        seed as usize,
        verbosity.detailed_verbosity(),
    )?;
    let dc_distances = dc_distances.expect("distances must be returned");

    // k_density-th NN distance per cell. fold(max) is robust to whichever order
    // the underlying ANN engine returns.
    let density_distances: Vec<f32> = dc_distances
        .iter()
        .map(|d| d.iter().copied().fold(0.0f32, f32::max).sqrt())
        .collect();

    let regions = classify_density_regions(&density_distances);

    Ok(DiffusionDensity {
        dcs: dc_mat,
        density_distances,
        regions,
    })
}

/// Compactness per metacell: average variance across DC dimensions over
/// the cells that constitute the metacell. Lower is better.
///
/// ### Params
///
/// * `dcs` - Diffusion components matrix (n_cells × n_dcs)
/// * `metacells` - Per-metacell cell indices
///
/// ### Returns
///
/// One compactness value per metacell. Empty metacells yield NaN.
pub fn compute_compactness(dcs: MatRef<f32>, metacells: &[Vec<usize>]) -> Vec<f32> {
    let d = dcs.ncols();

    metacells
        .par_iter()
        .map(|cells| {
            if cells.is_empty() {
                return f32::NAN;
            }
            let n = cells.len() as f32;

            let mut total_var = 0.0f32;
            for k in 0..d {
                let mut mean = 0.0f32;
                for &i in cells {
                    mean += dcs[(i, k)];
                }
                mean /= n;

                let mut var = 0.0f32;
                for &i in cells {
                    let diff = dcs[(i, k)] - mean;
                    var += diff * diff;
                }
                total_var += var / n;
            }

            total_var / d as f32
        })
        .collect()
}

/// Separation per metacell: Euclidean distance from the metacell's centroid
/// in DC space to the nearest other metacell's centroid. Higher is better.
///
/// ### Params
///
/// * `dcs` - Diffusion components matrix (n_cells × n_dcs)
/// * `metacells` - Per-metacell cell indices
///
/// ### Returns
///
/// One separation value per metacell. Empty metacells yield NaN.
pub fn compute_separation(dcs: MatRef<f32>, metacells: &[Vec<usize>]) -> Vec<f32> {
    let d = dcs.ncols();
    let k = metacells.len();

    let centroids: Vec<Vec<f32>> = metacells
        .par_iter()
        .map(|cells| {
            if cells.is_empty() {
                return Vec::new();
            }
            let n = cells.len() as f32;
            let mut centroid = vec![0.0f32; d];
            for &i in cells {
                for j in 0..d {
                    centroid[j] += dcs[(i, j)];
                }
            }
            for v in &mut centroid {
                *v /= n;
            }
            centroid
        })
        .collect();

    (0..k)
        .into_par_iter()
        .map(|i| {
            if centroids[i].is_empty() {
                return f32::NAN;
            }
            let mut min_dist_sq = f32::INFINITY;
            for j in 0..k {
                if i == j || centroids[j].is_empty() {
                    continue;
                }
                let dist_sq = euclidean_distance_static(&centroids[i], &centroids[j]);
                if dist_sq < min_dist_sq {
                    min_dist_sq = dist_sq;
                }
            }
            min_dist_sq.sqrt()
        })
        .collect()
}
