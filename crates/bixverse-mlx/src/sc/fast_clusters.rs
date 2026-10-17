//! MLX stage one for fast clustering.
//!
//! [`MlxKMeans`] runs the k-means through
//! [`ann_search_rs::mlx::k_means_mlx::k_means_clusters_mlx`] and the
//! cell-mediated top-m search as an exhaustive MLX kNN query of the cells
//! against the centroids. Everything after stage one is the CPU code in
//! [`bixverse_sc::sc_analysis::fast_clusters::fast_clusters`].

use ann_search_rs::mlx::k_means_mlx::k_means_clusters_mlx;
use ann_search_rs::utils::k_means_utils::KMeansTrainingParams;
use ann_search_rs::{build_exhaustive_index_mlx, query_exhaustive_index_mlx};
use faer::{Mat, MatRef};

use crate::prelude::*;
use bixverse_sc::sc_analysis::fast_clusters::Coarsener;

///////////////
// MlxKMeans //
///////////////

/// MLX k-means coarsener for fast clustering.
///
/// Full-batch Lloyd's on the default MLX GPU stream; every iteration runs,
/// there is no convergence check.
#[derive(Clone, Copy, Debug, Default)]
pub struct MlxKMeans {
    /// k-means parameters; `iters`, `init` and `balanced` are honoured.
    /// `None` takes the ann-search-rs defaults
    pub params: Option<KMeansTrainingParams>,
}

impl MlxKMeans {
    /// Generate a new MLX k-means coarsener.
    ///
    /// ### Params
    ///
    /// * `params` - k-means parameters, `None` for the defaults
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn new(params: Option<KMeansTrainingParams>) -> Self {
        Self { params }
    }
}

impl Coarsener for MlxKMeans {
    type Error = MlxErrors;

    /// k-means on MLX. Manhattan is rejected by ann-search-rs.
    fn coarsen(
        &self,
        data: MatRef<f32>,
        n_centroids: usize,
        dist: &str,
        seed: usize,
        verbose: usize,
    ) -> Result<(Mat<f32>, Vec<usize>), MlxErrors> {
        Ok(k_means_clusters_mlx(
            data,
            dist,
            n_centroids,
            self.params,
            seed,
            parse_verbosity_level(verbose).detailed_verbosity(),
        )?)
    }

    /// Exhaustive MLX kNN of the cells against the centroids.
    fn top_m(
        &self,
        data: MatRef<f32>,
        centroids: MatRef<f32>,
        m: usize,
        dist: &str,
    ) -> Result<Vec<u32>, MlxErrors> {
        let index = build_exhaustive_index_mlx(centroids, dist)?;
        let (idx, _) = query_exhaustive_index_mlx(data, &index, m, false, false)?;
        let mut out = Vec::with_capacity(idx.len() * m);
        for row in idx {
            out.extend(row.into_iter().map(|c| c as u32));
        }
        Ok(out)
    }
}
