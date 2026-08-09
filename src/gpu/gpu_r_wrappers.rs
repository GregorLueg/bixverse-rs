//! Contains R-specific functions for GPU-accelerated parts of this crate.

use extendr_api::*;
use std::collections::HashMap;

use crate::gpu::ml::k_means_gpu::KMeansGpuParams;
#[cfg(feature = "single-cell")]
use crate::gpu::sc_gpu::fast_clusters_gpu::FastLouvainParamsGpu;
#[cfg(feature = "single-cell")]
use crate::gpu::sc_gpu::harmony_gpu::HarmonyParamsV2Gpu;
use crate::ml::clustering::k_means::parse_kmeans_init;
#[cfg(feature = "single-cell")]
use crate::single_cell::sc_processing::knn::KnnParams;

/////////////////////
// KMeansGpuParams //
/////////////////////

impl KMeansGpuParams {
    /// Parse the [KMeansGpuParams] from a list
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list to parse
    ///
    /// ### Returns
    ///
    /// The [KMeansGpuParams] populated by the R list.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params_list: HashMap<&str, Robj> = r_list.try_into()?;

        let iters = params_list
            .get("k_means_iter")
            .and_then(|v| v.as_integer())
            .unwrap_or(50) as usize;

        let init = params_list
            .get("k_means_init")
            .and_then(|v| v.as_str())
            .and_then(parse_kmeans_init);

        let fixed = params_list
            .get("fixed")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let quantise_to_f16 = params_list
            .get("quantise")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(Self::new(iters, init, fixed, quantise_to_f16))
    }
}

//////////////////////////
// FastLouvainParamsGpu //
//////////////////////////

#[cfg(feature = "single-cell")]
impl FastLouvainParamsGpu {
    /// Generate [FastLouvainParamsGpu] from an R list.
    ///
    /// Should values not be found within the List, the parameters will default
    /// to the values defined in `FastLouvainParamsGpu::default()`. Field names
    /// mirror the CPU `FastLouvainParams::from_r_list`, with the k-means block
    /// coming from the GPU keys (`k_means_iter`, `k_means_init`, `fixed`,
    /// `quantise`) instead of the CPU ones.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the fast Louvain parameters.
    ///
    /// ### Returns
    ///
    /// The [FastLouvainParamsGpu] with all parameters set.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let knn_params = KnnParams::from_r_list(r_list.clone())?;
        let kmeans_params = Some(KMeansGpuParams::from_r_list(r_list.clone())?);

        let defaults = Self::default();
        let params: HashMap<&str, Robj> = r_list.try_into()?;

        let n_centroids = params
            .get("n_centroids")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.n_centroids);
        let same_weight = params
            .get("same_weight")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.same_weight);
        let full_snn = params
            .get("full_snn")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.full_snn);
        let pruning = params
            .get("pruning")
            .and_then(|v| v.as_real())
            .map(|v| v as f32);
        let snn_similarity = std::string::String::from(
            params
                .get("snn_similarity")
                .and_then(|v| v.as_str())
                .unwrap_or(&defaults.snn_similarity),
        );
        let louvain_iters = params
            .get("louvain_iters")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.louvain_iters);
        let multi_level_louvain = params
            .get("multi_level_louvain")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.multi_level_louvain);

        Ok(Self {
            n_centroids,
            kmeans_params,
            knn_params,
            same_weight,
            full_snn,
            pruning,
            snn_similarity,
            louvain_iters,
            multi_level_louvain,
        })
    }
}

////////////////////////
// HarmonyParamsV2Gpu //
////////////////////////

#[cfg(feature = "single-cell")]
impl HarmonyParamsV2Gpu {
    /// Generate HarmonyParamsV2Gpu from an R list.
    ///
    /// Should values not be found within the List, the parameters will default
    /// to the values defined in `HarmonyParamsV2Gpu::default()`. The
    /// `kmeans_params` field is populated from the same list, with
    /// GPU-Harmony-specific defaults (`iters = 30`, `fixed = false`).
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the Harmony parameters.
    ///
    /// ### Returns
    ///
    /// The `HarmonyParamsV2Gpu` with all parameters set.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let defaults = Self::default();
        let params_list: HashMap<&str, Robj> = r_list.try_into()?;

        let k = params_list
            .get("k")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.k);
        let sigma = params_list
            .get("sigma")
            .and_then(|v| v.as_real_vector())
            .map(|v| v.iter().map(|&x| x as f32).collect())
            .unwrap_or(defaults.sigma);
        let theta = params_list
            .get("theta")
            .and_then(|v| v.as_real_vector())
            .map(|v| v.iter().map(|&x| x as f32).collect())
            .unwrap_or(defaults.theta);
        let lambda = params_list
            .get("lambda")
            .and_then(|v| v.as_real_vector())
            .map(|v| v.iter().map(|&x| x as f32).collect())
            .unwrap_or(defaults.lambda);
        let max_iter_kmeans = params_list
            .get("max_iter_kmeans")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.max_iter_kmeans);
        let max_iter_harmony = params_list
            .get("max_iter_harmony")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.max_iter_harmony);
        let epsilon_kmeans = params_list
            .get("epsilon_kmeans")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.epsilon_kmeans);
        let epsilon_harmony = params_list
            .get("epsilon_harmony")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.epsilon_harmony);
        let window_size = params_list
            .get("window_size")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.window_size);
        let alpha = params_list
            .get("alpha")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.alpha);
        let tau = params_list
            .get("tau")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.tau);
        let batch_proportion_cutoff = params_list
            .get("batch_proportion_cutoff")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.batch_proportion_cutoff);
        let use_dynamic_lambda = params_list
            .get("use_dynamic_lambda")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.use_dynamic_lambda);
        let csr_cube_count = params_list
            .get("csr_cube_count")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.csr_cube_count);

        let kmeans_iters = params_list
            .get("k_means_iter")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(30);
        let kmeans_init = params_list
            .get("k_means_init")
            .and_then(|v| v.as_str())
            .and_then(parse_kmeans_init);
        let kmeans_fixed = params_list
            .get("fixed")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let kmeans_quantise = params_list
            .get("quantise")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let kmeans_params = Some(KMeansGpuParams::new(
            kmeans_iters,
            kmeans_init,
            kmeans_fixed,
            kmeans_quantise,
        ));

        Ok(Self {
            k,
            sigma,
            theta,
            lambda,
            max_iter_kmeans,
            max_iter_harmony,
            epsilon_kmeans,
            epsilon_harmony,
            window_size,
            alpha,
            tau,
            batch_proportion_cutoff,
            use_dynamic_lambda,
            csr_cube_count,
            kmeans_params,
        })
    }
}
