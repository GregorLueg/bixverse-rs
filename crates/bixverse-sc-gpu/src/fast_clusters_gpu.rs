//! GPU stage one for fast clustering.
//!
//! [`GpuKMeans`] runs the k-means on the device via
//! [`ann_search_rs::gpu::k_means_gpu::k_means_clusters_gpu`], and the
//! cell-mediated top-m search as an exhaustive GPU kNN query of the cells
//! against the centroids. The centroid graph and the community detection are
//! the CPU stages of [`bixverse_sc::sc_analysis::fast_clusters::fast_clusters`].

use ann_search_rs::gpu::k_means_gpu::{KMeansGpuParams, k_means_clusters_gpu};
use ann_search_rs::{build_exhaustive_index_gpu, query_exhaustive_index_gpu};
use cubecl::Device;
use faer::{Mat, MatRef};

use crate::prelude::*;
use ann_search_rs::prelude::EMeansParams;
use bixverse_core::ml::clustering::k_means::{parse_kmeans_algorithm, parse_kmeans_init};
use bixverse_sc::sc_analysis::fast_clusters::Coarsener;
use serde::Deserialize;

////////////
// Params //
////////////

/// The k-means keys of the R `params_sc_fast_cluster_gpu()` and
/// `params_kmeans_gpu()` lists, as `KMeansGpuParams::from_r_list` reads them;
/// missing keys take its defaults.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct KMeansGpuSpec {
    /// `"emeans"` or `"lloyd"`, see `parse_kmeans_algorithm`
    pub k_means_algorithm: String,
    /// e-means epoch budget, ignored for Lloyd's
    pub e_means_epochs: f64,
    /// Lloyd's iterations
    pub k_means_iter: usize,
    /// Optional initialisation, see `parse_kmeans_init`
    pub k_means_init: Option<String>,
    /// Run every iteration without a convergence check
    pub fixed: bool,
    /// Hold the data buffer at fp16 on the device
    pub quantise: bool,
}

impl Default for KMeansGpuSpec {
    fn default() -> Self {
        Self {
            k_means_algorithm: "emeans".to_string(),
            e_means_epochs: EMeansParams::default().epochs,
            k_means_iter: 50,
            k_means_init: None,
            fixed: true,
            quantise: false,
        }
    }
}

impl KMeansGpuSpec {
    /// Check the initialisation and assemble the GPU k-means parameters.
    ///
    /// ### Returns
    ///
    /// The [`KMeansGpuParams`], or an error naming an unknown algorithm or
    /// initialisation.
    pub fn into_params(self) -> Result<KMeansGpuParams, String> {
        let algorithm = parse_kmeans_algorithm(&self.k_means_algorithm, self.e_means_epochs)
            .ok_or_else(|| format!("Invalid k-means algorithm: {}", self.k_means_algorithm))?;
        let init = self
            .k_means_init
            .map(|s| {
                parse_kmeans_init(&s).ok_or_else(|| format!("Invalid k-means initialisation: {s}"))
            })
            .transpose()?;
        Ok(
            KMeansGpuParams::new(self.k_means_iter, init, self.fixed, self.quantise)
                .with_algorithm(algorithm),
        )
    }
}

///////////////
// GpuKMeans //
///////////////

/// GPU k-means coarsener for fast clustering.
///
/// Full-batch Lloyd's on the device; there is no mini-batch path.
#[derive(Clone, Debug)]
pub struct GpuKMeans {
    /// GPU k-means parameters; `None` takes the ann-search-rs defaults
    pub params: Option<KMeansGpuParams>,
    /// CubeCL runtime device
    pub device: Device,
}

impl GpuKMeans {
    /// Generate a new GPU k-means coarsener.
    ///
    /// ### Params
    ///
    /// * `params` - GPU k-means parameters, `None` for the defaults
    /// * `device` - CubeCL runtime device
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn new(params: Option<KMeansGpuParams>, device: Device) -> Self {
        Self { params, device }
    }
}

impl Coarsener for GpuKMeans {
    type Error = ScGpuErrors;

    /// k-means on the device. Manhattan is rejected by the driver; an
    /// unparseable metric falls back to squared Euclidean there.
    fn coarsen(
        &self,
        data: MatRef<f32>,
        n_centroids: usize,
        dist: &str,
        seed: usize,
        verbose: usize,
    ) -> Result<(Mat<f32>, Vec<usize>), ScGpuErrors> {
        let (centroids, assignments) = k_means_clusters_gpu::<f32>(
            data,
            dist,
            n_centroids,
            self.params,
            seed,
            self.device.clone(),
            verbosity_from_flag(parse_verbosity_level(verbose).detailed_verbosity()),
        )?;
        Ok((
            centroids,
            assignments.into_iter().map(|c| c as usize).collect(),
        ))
    }

    /// Exhaustive GPU kNN of the cells against the centroids.
    fn top_m(
        &self,
        data: MatRef<f32>,
        centroids: MatRef<f32>,
        m: usize,
        dist: &str,
    ) -> Result<Vec<u32>, ScGpuErrors> {
        let index =
            build_exhaustive_index_gpu(centroids, dist, self.device.clone(), Verbosity::Quiet)?;
        let (idx, _) = query_exhaustive_index_gpu(data, &index, m, false, Verbosity::Quiet)?;
        let mut out = Vec::with_capacity(idx.len() * m);
        for row in idx {
            out.extend(row.into_iter().map(|c| c as u32));
        }
        Ok(out)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use bixverse_sc::sc_analysis::fast_clusters::*;
    use cubecl::wgpu::WgpuDevice;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn make_two_blobs(n_per: usize, dim: usize, separation: f32, seed: u64) -> Mat<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        Mat::from_fn(2 * n_per, dim, |i, _| {
            let centre = if i < n_per { 0.0 } else { separation };
            centre + rng.random_range(-0.3..0.3)
        })
    }

    fn device() -> Device {
        WgpuDevice::default().into()
    }

    /// The GPU top-m returns the same centroid sets as the CPU scan.
    #[test]
    fn test_gpu_top_m_matches_cpu() {
        let data = make_two_blobs(300, 8, 3.0, 1);
        let centroids = Mat::from_fn(40, 8, |i, j| data[(i * 15, j)] + 0.01);
        let gpu = GpuKMeans::new(None, device())
            .top_m(data.as_ref(), centroids.as_ref(), 5, "euclidean")
            .unwrap();
        let cpu = top_m_cpu(data.as_ref(), centroids.as_ref(), 5, "euclidean");
        let mut mismatch = 0;
        for (g, c) in gpu.as_chunks::<5>().0.iter().zip(cpu.as_chunks::<5>().0) {
            let (mut g, mut c) = (g.to_vec(), c.to_vec());
            g.sort_unstable();
            c.sort_unstable();
            mismatch += (g != c) as usize;
        }
        // f32 ties at the m-th boundary may legitimately differ
        assert!(mismatch <= 3, "{mismatch} rows differ");
    }

    /// The GPU coarsener separates two blobs on the kNN and the
    /// cell-mediated centroid graphs.
    #[test]
    fn test_fast_clusters_gpu_separates_blobs() {
        let n_per = 200;
        let data = make_two_blobs(n_per, 8, 15.0, 42);
        let coarsener = GpuKMeans::new(None, device());
        for graph in [
            CentroidGraph::Knn {
                k: 5,
                sym: bixverse_methods::graph::knn_graph::KnnSymmetrisation::Knn { binary: false },
            },
            CentroidGraph::CellMediated { n_anchors: 5 },
        ] {
            let params = FastClusterParams {
                n_centroids: 30,
                graph,
                ..FastClusterParams::default()
            };
            let res = fast_clusters(data.as_ref(), &coarsener, &params, &[1.0], 1, 0, 0).unwrap();
            let labels = &res.per_resolution[0].labels;
            let n_labels = labels.iter().max().unwrap() + 1;
            for l in 0..n_labels {
                let in_a = labels[..n_per].iter().filter(|&&x| x == l).count();
                let in_b = labels[n_per..].iter().filter(|&&x| x == l).count();
                assert!(in_a == 0 || in_b == 0, "community {l} spans both blobs");
            }
        }
    }
}
