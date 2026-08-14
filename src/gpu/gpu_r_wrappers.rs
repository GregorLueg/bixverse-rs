//! Contains R-specific functions for GPU-accelerated parts of this crate.

use extendr_api::*;
use std::collections::HashMap;

use crate::gpu::ml::k_means_gpu::KMeansGpuParams;
#[cfg(feature = "single-cell")]
use crate::gpu::sc_gpu::fast_clusters_gpu::FastLouvainParamsGpu;
#[cfg(feature = "single-cell")]
use crate::gpu::sc_gpu::harmony_gpu::HarmonyParamsV2Gpu;
#[cfg(feature = "single-cell")]
use crate::gpu::sc_gpu::knn_gpu::KnnParamsGpu;
#[cfg(feature = "single-cell")]
use crate::gpu::sc_gpu::scrublet_gpu::{ScrubletKnnBackend, ScrubletParamsGpu};
use crate::ml::clustering::k_means::parse_kmeans_init;
#[cfg(feature = "single-cell")]
use crate::single_cell::sc_processing::knn::KnnParams;
use crate::utils::r_rust_interface::r_list_to_map;
#[cfg(feature = "single-cell")]
use crate::utils::r_rust_interface::{r_list_count, r_list_count_allow_zero};

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
        let params_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;

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
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;

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
        let params_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;

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

//////////////////
// KnnParamsGpu //
//////////////////

#[cfg(feature = "single-cell")]
impl KnnParamsGpu {
    /// Generate [KnnParamsGpu] from an R list.
    ///
    /// Reads the same flattened list `KnnParams::from_r_list` reads, but only
    /// the five keys the GPU indices understand. Missing keys fall back to
    /// [`KnnParamsGpu::default()`].
    ///
    /// `k` goes through [r_list_count_allow_zero]: zero is a legitimate value
    /// here, read downstream by `adjusted_k` as "derive it from the data".
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the kNN parameters.
    ///
    /// ### Returns
    ///
    /// The [KnnParamsGpu] with all parameters set.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let defaults = Self::default();
        let params_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let knn_method = std::string::String::from(
            params_list
                .get("knn_method")
                .and_then(|v| v.as_str())
                .unwrap_or(&defaults.knn_method),
        );
        let ann_dist = std::string::String::from(
            params_list
                .get("ann_dist")
                .and_then(|v| v.as_str())
                .unwrap_or(&defaults.ann_dist),
        );
        let k = r_list_count_allow_zero(&params_list, "k")?.unwrap_or(defaults.k);
        let n_list = r_list_count(&params_list, "n_list")?;
        let n_probe = r_list_count(&params_list, "n_probe")?;

        Ok(Self {
            knn_method,
            ann_dist,
            k,
            n_list,
            n_probe,
        })
    }
}

///////////////////////
// ScrubletParamsGpu //
///////////////////////

#[cfg(feature = "single-cell")]
impl ScrubletParamsGpu {
    /// Generate [ScrubletParamsGpu] from an R list.
    ///
    /// Field names mirror `ScrubletParams::from_r_list`, minus `random_svd`
    /// (the GPU SVD is always randomised), plus one key the CPU list has no
    /// need for: `knn_backend`.
    ///
    /// `knn_backend` is the only thing that can pick the
    /// [`ScrubletKnnBackend`] arm. Both backends share `k`, `knn_method`,
    /// `ann_dist`, `n_list` and `n_probe`, and `"exhaustive"` / `"ivf"` are
    /// legal `knn_method` values on either side, so the method string carries
    /// no information about which index was meant. Absent, the GPU arm is
    /// taken. Anything other than `"gpu"` or `"cpu"` is an error rather than a
    /// silent fallback: a typo should not quietly change which index runs.
    ///
    /// `n_bins_hist` is read first and `n_bins_histogram` second, so a list
    /// built by `bixverse::params_scrublet()` lands on the right value too.
    /// Keep both branches, they are not redundant.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the GPU Scrublet parameters.
    ///
    /// ### Returns
    ///
    /// The [ScrubletParamsGpu] with all parameters set.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let defaults = Self::default();
        let params_list: HashMap<&str, Robj> = r_list_to_map(r_list.clone())?;

        let backend = params_list
            .get("knn_backend")
            .and_then(|v| v.as_str())
            .unwrap_or("gpu")
            .to_lowercase();

        let knn_params = match backend.as_str() {
            "gpu" => ScrubletKnnBackend::Gpu(KnnParamsGpu::from_r_list(r_list)?),
            "cpu" => ScrubletKnnBackend::Cpu(KnnParams::from_r_list(r_list)?),
            other => {
                return Err(Error::Other(format!(
                    "Unknown `knn_backend`: '{other}'. Expected 'gpu' or 'cpu'."
                )));
            }
        };

        // -- processing --
        let log_transform = params_list
            .get("log_transform")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.log_transform);
        let mean_center = params_list
            .get("mean_center")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.mean_center);
        let normalise_variance = params_list
            .get("normalise_variance")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.normalise_variance);
        let target_size = params_list
            .get("target_size")
            .and_then(|v| v.as_real())
            .map(|x| x as f32);

        // -- hvg --
        let min_gene_var_pctl = params_list
            .get("min_gene_var_pctl")
            .and_then(|v| v.as_real())
            .map(|x| x as f32)
            .unwrap_or(defaults.min_gene_var_pctl);
        let hvg_method = std::string::String::from(
            params_list
                .get("hvg_method")
                .and_then(|v| v.as_str())
                .unwrap_or(&defaults.hvg_method),
        );
        let loess_span = params_list
            .get("loess_span")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.loess_span);
        let clip_max = params_list
            .get("clip_max")
            .and_then(|v| v.as_real())
            .map(|x| x as f32);
        let binning_strategy = std::string::String::from(
            params_list
                .get("binning_strategy")
                .and_then(|v| v.as_str())
                .unwrap_or(&defaults.binning_strategy),
        );
        let n_bins = r_list_count(&params_list, "n_bins")?.unwrap_or(defaults.n_bins);

        // -- pca --
        let no_pcs = r_list_count(&params_list, "no_pcs")?.unwrap_or(defaults.no_pcs);

        // -- scrublet --
        let sim_doublet_ratio = params_list
            .get("sim_doublet_ratio")
            .and_then(|v| v.as_real())
            .map(|x| x as f32)
            .unwrap_or(defaults.sim_doublet_ratio);
        let expected_doublet_rate = params_list
            .get("expected_doublet_rate")
            .and_then(|v| v.as_real())
            .map(|x| x as f32)
            .unwrap_or(defaults.expected_doublet_rate);
        let stdev_doublet_rate = params_list
            .get("stdev_doublet_rate")
            .and_then(|v| v.as_real())
            .map(|x| x as f32)
            .unwrap_or(defaults.stdev_doublet_rate);
        let n_bins_hist = r_list_count(&params_list, "n_bins_hist")?
            .or(r_list_count(&params_list, "n_bins_histogram")?)
            .unwrap_or(defaults.n_bins_hist);
        let manual_threshold = params_list
            .get("manual_threshold")
            .and_then(|v| v.as_real())
            .map(|x| x as f32);

        Ok(Self {
            log_transform,
            mean_center,
            normalise_variance,
            target_size,
            min_gene_var_pctl,
            hvg_method,
            loess_span,
            clip_max,
            binning_strategy,
            n_bins,
            no_pcs,
            sim_doublet_ratio,
            expected_doublet_rate,
            stdev_doublet_rate,
            n_bins_hist,
            manual_threshold,
            knn_params,
        })
    }
}
