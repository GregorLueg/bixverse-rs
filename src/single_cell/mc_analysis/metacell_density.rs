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
    let mut kernel = compute_diffusion_kernel(knn_indices, knn_distances)?;

    if verbosity.normal_verbosity() {
        println!("Computing top {} diffusion components...", n_dcs);
    }
    let (evals, evecs) = diffusion_map_from_kernel(&mut kernel, n_dcs + 1, seed, None)?;
    let dcs = determine_multiscale_space(&evals, &evecs, Some(n_dcs + 1));

    // The solver caps the pair count at the matrix dimension, so a metacell set
    // smaller than the requested component count yields a narrower embedding
    // than asked for, and an empty one when there is only a single metacell.
    let dim = dcs.first().map_or(0, Vec::len);
    if dim == 0 {
        return Err(BixverseErrors::InvalidArgument(
            "Diffusion density: too few metacells to build a diffusion embedding".to_string(),
        ));
    }

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
        .map(|d| d.iter().copied().fold(0.0f32, f32::max))
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
/// One separation value per metacell. Empty metacells yield NaN. A lone
/// non-empty metacell has no other centroid to measure against and yields
/// positive infinity.
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Four cells on a line in a two-dimensional DC space: x = 0, 2, 6, 10,
    /// y = 0 throughout. Every metric below is hand-derivable from that.
    fn line_dcs() -> Mat<f32> {
        let xs = [0.0f32, 2.0, 6.0, 10.0];
        Mat::<f32>::from_fn(4, 2, |i, j| if j == 0 { xs[i] } else { 0.0 })
    }

    /////////////////
    // Compactness //
    /////////////////

    /// Average per-DC variance, worked by hand: metacell {0, 2} has mean 1 and
    /// variance 1 on x, 0 on y, so (1 + 0) / 2 = 0.5; metacell {6, 10} has mean
    /// 8 and variance 4, so (4 + 0) / 2 = 2.
    #[test]
    fn compute_compactness_averages_variance_across_dcs() {
        let dcs = line_dcs();
        let metacells = vec![vec![0, 1], vec![2, 3]];

        let compactness = compute_compactness(dcs.as_ref(), &metacells);

        assert_relative_eq!(compactness[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(compactness[1], 2.0, epsilon = 1e-6);
    }

    /// A single cell has zero spread in every dimension.
    #[test]
    fn compute_compactness_is_zero_for_a_singleton_metacell() {
        let dcs = line_dcs();
        let compactness = compute_compactness(dcs.as_ref(), &[vec![3]]);

        assert_relative_eq!(compactness[0], 0.0, epsilon = 1e-6);
    }

    /// The docstring promises NaN for empty metacells, and the position in the
    /// output vector has to be kept rather than dropped.
    #[test]
    fn compute_compactness_yields_nan_for_empty_metacells() {
        let dcs = line_dcs();
        let metacells = vec![vec![0, 1], vec![], vec![2, 3]];

        let compactness = compute_compactness(dcs.as_ref(), &metacells);

        assert_eq!(compactness.len(), 3);
        assert!(compactness[1].is_nan());
        assert!(compactness[0].is_finite() && compactness[2].is_finite());
    }

    ////////////////
    // Separation //
    ////////////////

    /// Centroids sit at x = 1 and x = 8, so the nearest-neighbour centroid
    /// distance is 7 for both. Also pins that the squared distance from
    /// `euclidean_distance_static` is square-rooted before it is returned.
    #[test]
    fn compute_separation_returns_the_nearest_centroid_distance() {
        let dcs = line_dcs();
        let metacells = vec![vec![0, 1], vec![2, 3]];

        let separation = compute_separation(dcs.as_ref(), &metacells);

        assert_relative_eq!(separation[0], 7.0, epsilon = 1e-5);
        assert_relative_eq!(separation[1], 7.0, epsilon = 1e-5);
    }

    /// With three metacells the nearest one wins, not the mean or the last.
    /// Centroids: 1, 8 and 10, so the middle one is 2 away from its right
    /// neighbour and 7 from its left.
    #[test]
    fn compute_separation_picks_the_closest_other_centroid() {
        let dcs = line_dcs();
        let metacells = vec![vec![0, 1], vec![2, 3], vec![3]];

        let separation = compute_separation(dcs.as_ref(), &metacells);

        assert_relative_eq!(separation[0], 7.0, epsilon = 1e-5);
        assert_relative_eq!(separation[1], 2.0, epsilon = 1e-5);
        assert_relative_eq!(separation[2], 2.0, epsilon = 1e-5);
    }

    /// Empty metacells yield NaN and are skipped as comparison partners, so
    /// the remaining pair still measures against each other.
    #[test]
    fn compute_separation_yields_nan_for_empty_metacells() {
        let dcs = line_dcs();
        let metacells = vec![vec![0, 1], vec![], vec![2, 3]];

        let separation = compute_separation(dcs.as_ref(), &metacells);

        assert!(separation[1].is_nan());
        assert_relative_eq!(separation[0], 7.0, epsilon = 1e-5);
        assert_relative_eq!(separation[2], 7.0, epsilon = 1e-5);
    }

    /// A lone metacell has nothing to compare against, so `min_dist_sq` stays
    /// at its infinite initial value. Infinity, not NaN.
    #[test]
    fn compute_separation_yields_infinity_for_a_single_metacell() {
        let dcs = line_dcs();
        let separation = compute_separation(dcs.as_ref(), &[vec![0, 1]]);

        assert!(separation[0].is_infinite() && separation[0].is_sign_positive());
    }

    ////////////////////////
    // Density regions //
    ////////////////////////

    /// Quartile split on eight distances: the two smallest are High, the two
    /// largest Low, the middle four Mid, and the labels follow the original
    /// positions rather than the sorted order.
    #[test]
    fn classify_density_regions_splits_on_quartiles() {
        let dist = [5.0f32, 1.0, 7.0, 0.0, 3.0, 6.0, 2.0, 4.0];

        let regions = classify_density_regions(&dist);

        use DensityRegion::*;
        assert_eq!(regions, vec![Mid, High, Low, High, Mid, Low, Mid, Mid]);
    }

    /// With fewer than four inputs the High bucket collapses to nothing while
    /// the Low bucket does not, which is the asymmetry a caller has to know.
    #[test]
    fn classify_density_regions_handles_tiny_inputs() {
        let regions = classify_density_regions(&[1.0, 0.0]);

        assert_eq!(regions, vec![DensityRegion::Low, DensityRegion::Mid]);
        assert!(classify_density_regions(&[]).is_empty());
    }

    ///////////////////////
    // Diffusion density //
    ///////////////////////

    /// A single cell cannot carry a diffusion embedding: the solver returns at
    /// most one eigenpair, the multiscale space comes back zero-wide, and the
    /// guard has to fire before the kNN search reads it.
    #[test]
    fn compute_diffusion_density_rejects_a_zero_width_embedding() {
        let knn_indices = vec![vec![0]];
        let knn_distances = vec![vec![0.0f32]];

        let result = compute_diffusion_density(
            &knn_indices,
            &knn_distances,
            10,
            5,
            &KnnParams::default(),
            42,
            0,
        );

        assert!(matches!(result, Err(BixverseErrors::InvalidArgument(_))));
    }
}
