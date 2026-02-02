//! Contains R-specific functions for single-cell data processing that need
//! the extendr interface.

use extendr_api::*;

use crate::single_cell::sc_data::data_io::MinCellQuality;
use crate::single_cell::sc_processing::doublet_detection::BoostParams;
use crate::single_cell::sc_processing::knn::KnnParams;
use crate::single_cell::sc_processing::scrublet::ScrubletParams;

//////////////////
// Cell quality //
//////////////////

impl MinCellQuality {
    /// Generate the MinCellQuality params from an R list
    ///
    /// or default to sensible defaults
    ///
    /// ### Params
    ///
    /// * `r_list` - R list with the parameters
    ///
    /// ### Returns
    ///
    /// Self with the specified parameters.
    pub fn from_r_list(r_list: List) -> Self {
        let min_qc = r_list.into_hashmap();

        let min_unique_genes = min_qc
            .get("min_unique_genes")
            .and_then(|v| v.as_integer())
            .unwrap_or(100) as usize;

        let min_lib_size = min_qc
            .get("min_lib_size")
            .and_then(|v| v.as_integer())
            .unwrap_or(250) as usize;

        let min_cells = min_qc
            .get("min_cells")
            .and_then(|v| v.as_integer())
            .unwrap_or(10) as usize;

        let target_size = min_qc
            .get("target_size")
            .and_then(|v| v.as_real())
            .unwrap_or(1e5) as f32;

        MinCellQuality {
            min_unique_genes,
            min_lib_size,
            min_cells,
            target_size,
        }
    }
}

///////////////
// KnnParams //
///////////////

impl KnnParams {
    /// Generate KnnParams from an R list
    ///
    /// Should values not be found within the List, the parameters will default
    /// to sensible defaults based on heuristics.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the kNN parameters.
    ///
    /// ### Returns
    ///
    /// The `KnnParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Self {
        let params_list = r_list.into_hashmap();

        // general
        let knn_method = std::string::String::from(
            params_list
                .get("knn_method")
                .and_then(|v| v.as_str())
                .unwrap_or("annoy"),
        );

        let ann_dist = std::string::String::from(
            params_list
                .get("ann_dist")
                .and_then(|v| v.as_str())
                .unwrap_or("cosine"),
        );

        let k = params_list
            .get("k")
            .and_then(|v| v.as_integer())
            .unwrap_or(15) as usize;

        // annoy
        let n_tree = params_list
            .get("n_trees")
            .and_then(|v| v.as_integer())
            .unwrap_or(50) as usize;

        let search_budget = params_list
            .get("search_budget")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        // nn descent
        let diversify_prob = params_list
            .get("diversify_prob")
            .and_then(|v| v.as_real())
            .unwrap_or(0.0) as f32;

        let delta = params_list
            .get("delta")
            .and_then(|v| v.as_real())
            .unwrap_or(0.001) as f32;

        let ef_budget = params_list
            .get("ef_budget")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        // hnsw
        let m = params_list
            .get("m")
            .and_then(|v| v.as_integer())
            .unwrap_or(16) as usize;

        let ef_construction = params_list
            .get("ef_construction")
            .and_then(|v| v.as_integer())
            .unwrap_or(200) as usize;

        let ef_search = params_list
            .get("ef_search")
            .and_then(|v| v.as_integer())
            .unwrap_or(100) as usize;

        Self {
            knn_method,
            ann_dist,
            k,
            n_tree,
            search_budget,
            ef_budget,
            diversify_prob,
            delta,
            m,
            ef_construction,
            ef_search,
        }
    }
}

//////////////
// Scrublet //
//////////////

impl ScrubletParams {
    /// Generate ScrubletParams from an R list
    ///
    /// Should values not be found within the List, the parameters will default
    /// to sensible defaults based on Scrublet's original implementation.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the Scrublet parameters.
    ///
    /// ### Returns
    ///
    /// The `ScrubletParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Self {
        let knn_params = KnnParams::from_r_list(r_list.clone());

        let scrublet_list = r_list.into_hashmap();

        // General params
        let log_transform = scrublet_list
            .get("log_transform")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let mean_center = scrublet_list
            .get("mean_center")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let normalise_variance = scrublet_list
            .get("normalise_variance")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let target_size = scrublet_list
            .get("target_size")
            .and_then(|v| v.as_real())
            .map(|x| x as f32);

        // HVG detection parameters
        let min_gene_var_pctl = scrublet_list
            .get("min_gene_var_pctl")
            .and_then(|v| v.as_real())
            .unwrap_or(0.85) as f32;

        let hvg_method = std::string::String::from(
            scrublet_list
                .get("hvg_method")
                .and_then(|v| v.as_str())
                .unwrap_or("vst"),
        );

        let loess_span = scrublet_list
            .get("loess_span")
            .and_then(|v| v.as_real())
            .unwrap_or(0.3);

        let clip_max = scrublet_list
            .get("clip_max")
            .and_then(|v| v.as_real())
            .map(|x| x as f32);

        // Doublet simulation parameters
        let sim_doublet_ratio = scrublet_list
            .get("sim_doublet_ratio")
            .and_then(|v| v.as_real())
            .unwrap_or(2.0) as f32;

        let expected_doublet_rate = scrublet_list
            .get("expected_doublet_rate")
            .and_then(|v| v.as_real())
            .unwrap_or(0.1) as f32;

        let stdev_doublet_rate = scrublet_list
            .get("stdev_doublet_rate")
            .and_then(|v| v.as_real())
            .unwrap_or(0.02) as f32;

        // Doublet calling parameters
        let n_bins = scrublet_list
            .get("n_bins")
            .and_then(|v| v.as_integer())
            .unwrap_or(50) as usize;

        let manual_threshold = scrublet_list
            .get("manual_threshold")
            .and_then(|v| v.as_real())
            .map(|x| x as f32);

        // PCA parameters
        let no_pcs = scrublet_list
            .get("no_pcs")
            .and_then(|v| v.as_integer())
            .unwrap_or(30) as usize;

        let random_svd = scrublet_list
            .get("random_svd")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        Self {
            // norm
            log_transform,
            normalise_variance,
            mean_center,
            target_size,
            // hvg
            min_gene_var_pctl,
            hvg_method,
            loess_span,
            clip_max,
            // doublet simulation/detection
            sim_doublet_ratio,
            expected_doublet_rate,
            stdev_doublet_rate,
            n_bins,
            manual_threshold,
            // pca
            no_pcs,
            random_svd,
            // knn
            knn_params,
        }
    }
}

impl BoostParams {
    /// Generate BoostParams from an R list
    ///
    /// Should values not be found within the List, the parameters will default
    /// to sensible defaults based on the Boost algorithm.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the Boost parameters.
    ///
    /// ### Returns
    ///
    /// The `BoostParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Self {
        let knn_params = KnnParams::from_r_list(r_list.clone());

        let params_list = r_list.into_hashmap();

        // norm parameters
        let log_transform = params_list
            .get("log_transform")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let mean_center = params_list
            .get("mean_center")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let normalise_variance = params_list
            .get("normalise_variance")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let target_size = params_list
            .get("target_size")
            .and_then(|v| v.as_real())
            .map(|x| x as f32);

        // hvg
        let min_gene_var_pctl = params_list
            .get("min_gene_var_pctl")
            .and_then(|v| v.as_real())
            .unwrap_or(0.85) as f32;

        let hvg_method = std::string::String::from(
            params_list
                .get("hvg_method")
                .and_then(|v| v.as_str())
                .unwrap_or("vst"),
        );

        let loess_span = params_list
            .get("loess_span")
            .and_then(|v| v.as_real())
            .unwrap_or(0.3);

        let clip_max = params_list
            .get("clip_max")
            .and_then(|v| v.as_real())
            .map(|x| x as f32);

        // doublet detection params
        let boost_rate = params_list
            .get("boost_rate")
            .and_then(|v| v.as_real())
            .unwrap_or(0.25) as f32;

        let replace = params_list
            .get("replace")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let resolution = params_list
            .get("resolution")
            .and_then(|v| v.as_real())
            .unwrap_or(1.0) as f32;

        let louvain_iters = params_list
            .get("louvain_iters")
            .and_then(|v| v.as_integer())
            .unwrap_or(10) as usize;

        let n_iters = params_list
            .get("n_iters")
            .and_then(|v| v.as_integer())
            .unwrap_or(10) as usize;

        // pca
        let no_pcs = params_list
            .get("no_pcs")
            .and_then(|v| v.as_integer())
            .unwrap_or(30) as usize;

        let random_svd = params_list
            .get("random_svd")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let p_thresh = params_list
            .get("p_thresh")
            .and_then(|v| v.as_real())
            .unwrap_or(1e-7) as f32;

        let voter_thresh = params_list
            .get("voter_thresh")
            .and_then(|v| v.as_real())
            .unwrap_or(0.9) as f32;

        Self {
            log_transform,
            mean_center,
            normalise_variance,
            target_size,
            min_gene_var_pctl,
            hvg_method,
            loess_span,
            clip_max,
            boost_rate,
            replace,
            no_pcs,
            random_svd,
            resolution,
            louvain_iters,
            n_iters,
            p_thresh,
            voter_thresh,
            knn_params,
        }
    }
}
