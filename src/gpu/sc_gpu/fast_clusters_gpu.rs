//! GPU-accelerated version of the fast Louvain clustering pipeline.
//!
//! Only stage one changes: the k-means coarsening runs on the device via
//! [`k_means_clusters_gpu`]. The centroid kNN, the optional sNN pass and the
//! Louvain runs all reuse the CPU stages from
//! [`crate::single_cell::sc_analysis::fast_clusters`]. That split is deliberate:
//! k-means over `n` cells is the only part that scales with the data, while
//! everything downstream operates on ~1000 centroids and is already cheap.

use cubecl::prelude::*;
use faer::MatRef;

use crate::gpu::ml::k_means_gpu::{KMeansGpuParams, k_means_clusters_gpu};
use crate::prelude::*;
use crate::single_cell::sc_analysis::fast_clusters::*;
use crate::single_cell::sc_processing::knn::KnnParams;

////////////
// Params //
////////////

/// Parameters for GPU fast Louvain clustering via k-means + kNN.
///
/// Mirrors [crate::single_cell::sc_analysis::fast_clusters::FastLouvainParams]
/// minus the mini-batch knobs (`batch_size`, `drift_threshold`, `lr_alpha`), as
/// the GPU k-means is full-batch Lloyd's, and with [KMeansGpuParams] in place of
/// the CPU parameter wrapper.
#[derive(Clone, Debug)]
pub struct FastLouvainParamsGpu {
    // -- k means --
    /// Number of k-means centroids.
    pub n_centroids: usize,
    /// GPU k-means parameters; `None` uses the k-means defaults.
    pub kmeans_params: Option<KMeansGpuParams>,

    // -- knn --
    /// [KnnParams] parameters applied to the centroids. `ann_dist` also drives
    /// the k-means distance and `k` is the number of neighbours per centroid.
    pub knn_params: KnnParams,
    /// Shall the weight of the kNN be set to 1.0 across or edges that have
    /// reverse edges be double counted.
    pub same_weight: bool,

    // -- snn --
    /// Shall the full sNN graph be generated (also from non-connected
    /// neighbours)
    pub full_snn: bool,
    /// Optional pruning threshold. If not provided, will default to
    /// `1 / ceil(0.8 * k)`.
    pub pruning: Option<f32>,
    /// String for the type of sNN similarity to use. One of `"jaccard"` or
    /// `"rank"`.
    pub snn_similarity: String,

    // -- louvain --
    /// Number of Louvain iterations.
    pub louvain_iters: usize,
    /// Shall multi-level Louvain be applied
    pub multi_level_louvain: bool,
}

impl FastLouvainParamsGpu {
    /// Generate a version of [FastLouvainParamsGpu] with sensible base
    /// parameters. Matches the CPU defaults where the fields overlap.
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn new() -> Self {
        Self {
            n_centroids: 1000,
            kmeans_params: None,
            knn_params: KnnParams::default(),
            same_weight: false,
            full_snn: false,
            pruning: None,
            snn_similarity: "jaccard".to_string(),
            louvain_iters: 10,
            multi_level_louvain: true,
        }
    }

    /// Extract the graph-construction settings for [`build_centroid_graph`].
    ///
    /// ### Returns
    ///
    /// The [CentroidGraphParams].
    pub fn graph_params(&self) -> CentroidGraphParams {
        CentroidGraphParams {
            knn_params: self.knn_params.clone(),
            same_weight: self.same_weight,
            full_snn: self.full_snn,
            pruning: self.pruning,
            snn_similarity: self.snn_similarity.clone(),
        }
    }
}

/// Default implementation for [FastLouvainParamsGpu]
impl Default for FastLouvainParamsGpu {
    fn default() -> Self {
        Self::new()
    }
}

/////////////
// Stage 1 //
/////////////

/// GPU k-means coarsening of the data.
///
/// The GPU sibling of
/// [crate::single_cell::sc_analysis::fast_clusters::fast_cluster_kmeans]. The
/// requested centroid count is clamped to `n_samples - 1` and the distance
/// metric comes from `params.knn_params.ann_dist`, so the coarsening and the
/// centroid kNN agree on the geometry. Manhattan is rejected by the k-means
/// driver; an unparseable metric string falls back to squared Euclidean there.
///
/// ### Params
///
/// * `data` - n_samples x n_features matrix.
/// * `params` - Pipeline parameters.
/// * `seed` - Seed for reproducibility.
/// * `device` - CubeCL runtime device.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// `(centroids of shape n_centroids x n_features, per-sample centroid index)`.
///
/// ### Errors
///
/// * Propagates the k-means dispatch, allocation and read-back errors.
pub fn fast_cluster_kmeans_gpu<R: Runtime>(
    data: MatRef<f32>,
    params: &FastLouvainParamsGpu,
    seed: usize,
    device: R::Device,
    verbose: usize,
) -> Result<(faer::Mat<f32>, Vec<usize>), BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let n_centroids = params.n_centroids.min(data.nrows() - 1);

    k_means_clusters_gpu::<f32, R>(
        data,
        &params.knn_params.ann_dist,
        n_centroids,
        params.kmeans_params,
        seed,
        device,
        verbosity.detailed_verbosity(),
    )
}

//////////
// Main //
//////////

/// Fast Louvain clustering on large data via GPU k-means coarsening.
///
/// Same pipeline as
/// [crate::single_cell::sc_analysis::fast_clusters::fast_louvain_clusters], but
/// stage one runs on the device. There is no `km_type` parameter: the GPU
/// k-means has no mini-batch path.
///
/// ### Params
///
/// * `data` - n_samples x n_features matrix.
/// * `resolutions` - Slice of resolutions to iterate over.
/// * `params` - Pipeline parameters.
/// * `run_snn` - Shall a shared nearest neighbour generation be run.
/// * `return_k_mean` - Shall the k-means centroids and assignments be returned
///   alongside the Louvain memberships.
/// * `seed` - Seed for reproducibility.
/// * `device` - CubeCL runtime device.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The [FastLouvainResults], `Single` variant.
///
/// ### Errors
///
/// * Propagates the k-means, kNN and Louvain errors.
#[allow(clippy::too_many_arguments)]
pub fn fast_louvain_clusters_gpu<R: Runtime>(
    data: MatRef<f32>,
    resolutions: &[f32],
    params: &FastLouvainParamsGpu,
    run_snn: bool,
    return_k_mean: bool,
    seed: usize,
    device: R::Device,
    verbose: usize,
) -> Result<FastLouvainResults, BixverseErrors> {
    let (centroids, assignments) =
        fast_cluster_kmeans_gpu::<R>(data, params, seed, device, verbose)?;

    let graph = build_centroid_graph(
        centroids.as_ref(),
        &params.graph_params(),
        run_snn,
        seed,
        verbose,
    )?;

    let results = louvain_over_resolutions(
        &graph,
        &assignments,
        resolutions,
        params.louvain_iters,
        params.multi_level_louvain,
        seed,
    )?;

    Ok(FastLouvainResults::single(
        results,
        centroids,
        assignments,
        return_k_mean,
    ))
}

/// Grid-search version of [`fast_louvain_clusters_gpu`] with stability metrics.
///
/// Builds the k-means -> kNN/sNN graph once, then runs Louvain across
/// `n_louvain_seeds` seeds for each resolution.
///
/// ### Params
///
/// * `data` - n_samples x n_features matrix.
/// * `resolutions` - Slice of resolutions to iterate over.
/// * `params` - Pipeline parameters.
/// * `run_snn` - Shall a shared nearest neighbour generation be run.
/// * `return_k_mean` - Shall the k-means centroids and assignments be returned
///   alongside the grid results.
/// * `seed` - Seed for k-means and kNN (the shared graph).
/// * `n_louvain_seeds` - Number of seeds to vary Louvain over. Must be at least
///   2 to produce ARI.
/// * `device` - CubeCL runtime device.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The [FastLouvainResults], `GridRes` variant, with one
/// [crate::single_cell::sc_analysis::fast_clusters::FastLouvainGridResult] per
/// resolution in the same order.
///
/// ### Errors
///
/// * Propagates the k-means, kNN and Louvain errors, plus `InvalidArgument` for
///   fewer than two Louvain seeds.
#[allow(clippy::too_many_arguments)]
pub fn fast_louvain_clusters_grid_gpu<R: Runtime>(
    data: MatRef<f32>,
    resolutions: &[f32],
    params: &FastLouvainParamsGpu,
    run_snn: bool,
    return_k_mean: bool,
    seed: usize,
    n_louvain_seeds: usize,
    device: R::Device,
    verbose: usize,
) -> Result<FastLouvainResults, BixverseErrors> {
    let (centroids, assignments) =
        fast_cluster_kmeans_gpu::<R>(data, params, seed, device, verbose)?;

    let graph = build_centroid_graph(
        centroids.as_ref(),
        &params.graph_params(),
        run_snn,
        seed,
        verbose,
    )?;

    let results = louvain_grid_over_resolutions(
        &graph,
        &assignments,
        resolutions,
        params.louvain_iters,
        params.multi_level_louvain,
        seed,
        n_louvain_seeds,
    )?;

    Ok(FastLouvainResults::grid(
        results,
        centroids,
        assignments,
        return_k_mean,
    ))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use either::Either;
    use faer::Mat;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashMap;

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    fn make_two_blobs(n_per: usize, dim: usize, separation: f32, seed: u64) -> Mat<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        Mat::from_fn(2 * n_per, dim, |i, _| {
            let centre = if i < n_per { 0.0 } else { separation };
            centre + rng.random_range(-0.3..0.3)
        })
    }

    fn default_params() -> FastLouvainParamsGpu {
        let mut knn = KnnParams::new();
        knn.knn_method = "exhaustive".to_string();
        knn.ann_dist = "euclidean".to_string();
        knn.k = 5;
        FastLouvainParamsGpu {
            n_centroids: 30,
            knn_params: knn,
            ..FastLouvainParamsGpu::default()
        }
    }

    fn majority(xs: &[usize]) -> usize {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &x in xs {
            *counts.entry(x).or_insert(0) += 1;
        }
        *counts.iter().max_by_key(|(_, c)| **c).unwrap().0
    }

    fn unwrap_single(r: FastLouvainResults) -> Vec<Vec<usize>> {
        match r.get_assignments() {
            Either::Left(v) => v,
            Either::Right(_) => panic!("expected Single variant"),
        }
    }

    fn unwrap_grid(r: FastLouvainResults) -> Vec<FastLouvainGridResult> {
        match r.get_assignments() {
            Either::Right(v) => v,
            Either::Left(_) => panic!("expected GridRes variant"),
        }
    }

    // Both blobs must land in distinct, near-pure communities.
    fn assert_blobs_separated(labels: &[usize], n_per: usize) {
        let (blob_a, blob_b) = labels.split_at(n_per);
        let maj_a = majority(blob_a);
        let maj_b = majority(blob_b);
        assert_ne!(maj_a, maj_b);

        let purity_a = blob_a.iter().filter(|&&l| l == maj_a).count() as f32 / n_per as f32;
        let purity_b = blob_b.iter().filter(|&&l| l == maj_b).count() as f32 / n_per as f32;
        assert!(purity_a > 0.9, "blob A purity {}", purity_a);
        assert!(purity_b > 0.9, "blob B purity {}", purity_b);
    }

    #[test]
    fn test_gpu_separated_blobs() {
        let Some(device) = try_device() else { return };
        let n_per = 200;
        let data = make_two_blobs(n_per, 8, 15.0, 42);
        let resolutions: Vec<f32> = vec![1.0];

        let labels = unwrap_single(
            fast_louvain_clusters_gpu::<WgpuRuntime>(
                data.as_ref(),
                &resolutions,
                &default_params(),
                false,
                false,
                0,
                device,
                0,
            )
            .unwrap(),
        );

        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].len(), 2 * n_per);
        assert_blobs_separated(&labels[0], n_per);
    }

    #[test]
    fn test_gpu_separated_blobs_snn() {
        let Some(device) = try_device() else { return };
        let n_per = 200;
        let data = make_two_blobs(n_per, 8, 15.0, 42);
        let resolutions: Vec<f32> = vec![1.0];

        // Jaccard pruning at k = 5 leaves the 15 centroids per blob too sparsely
        // connected and Louvain splits a blob in two. k = 10 is what the sNN path
        // wants at this centroid count.
        let mut params = default_params();
        params.knn_params.k = 10;

        let labels = unwrap_single(
            fast_louvain_clusters_gpu::<WgpuRuntime>(
                data.as_ref(),
                &resolutions,
                &params,
                true,
                false,
                0,
                device,
                0,
            )
            .unwrap(),
        );

        assert_blobs_separated(&labels[0], n_per);
    }

    #[test]
    fn test_gpu_deterministic_with_same_seed() {
        let Some(device) = try_device() else { return };
        let data = make_two_blobs(100, 5, 5.0, 42);
        let resolutions: Vec<f32> = vec![1.0];

        let run = |device: WgpuDevice| {
            unwrap_single(
                fast_louvain_clusters_gpu::<WgpuRuntime>(
                    data.as_ref(),
                    &resolutions,
                    &default_params(),
                    false,
                    false,
                    7,
                    device,
                    0,
                )
                .unwrap(),
            )
        };

        assert_eq!(run(device.clone()), run(device));
    }

    #[test]
    fn test_gpu_returns_kmeans_when_requested() {
        let Some(device) = try_device() else { return };
        let data = make_two_blobs(100, 5, 5.0, 42);
        let params = default_params();

        let res = fast_louvain_clusters_gpu::<WgpuRuntime>(
            data.as_ref(),
            &[1.0],
            &params,
            false,
            true,
            0,
            device,
            0,
        )
        .unwrap();

        let centroids = res.get_centroids().unwrap();
        assert_eq!(centroids.nrows(), params.n_centroids);
        assert_eq!(centroids.ncols(), data.ncols());
        assert_eq!(res.get_k_mean_clusters().unwrap().len(), data.nrows());
    }

    #[test]
    fn test_gpu_grid_shape_and_metrics() {
        let Some(device) = try_device() else { return };
        let n_per = 200;
        let data = make_two_blobs(n_per, 8, 15.0, 42);
        let resolutions: Vec<f32> = vec![0.5, 1.0];

        let results = unwrap_grid(
            fast_louvain_clusters_grid_gpu::<WgpuRuntime>(
                data.as_ref(),
                &resolutions,
                &default_params(),
                false,
                false,
                0,
                4,
                device,
                0,
            )
            .unwrap(),
        );

        assert_eq!(results.len(), resolutions.len());
        for (r, res) in results.iter().zip(resolutions.iter()) {
            assert_eq!(r.resolution, *res);
            assert_eq!(r.best_labels.len(), 2 * n_per);
            assert!(
                r.mean_ari >= -1.0 && r.mean_ari <= 1.0,
                "mean_ari {}",
                r.mean_ari
            );
            assert!(
                r.mean_conductance >= 0.0,
                "conductance {}",
                r.mean_conductance
            );
            assert!(r.mean_n_communities >= 1.0);
            assert_blobs_separated(&r.best_labels, n_per);
        }
    }

    #[test]
    fn test_gpu_grid_rejects_too_few_seeds() {
        let Some(device) = try_device() else { return };
        let data = make_two_blobs(100, 5, 5.0, 42);

        let result = fast_louvain_clusters_grid_gpu::<WgpuRuntime>(
            data.as_ref(),
            &[1.0],
            &default_params(),
            false,
            false,
            0,
            1,
            device,
            0,
        );

        assert!(result.is_err());
    }
}
