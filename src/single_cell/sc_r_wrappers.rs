//! Contains R-specific functions for single-cell data processing that need
//! the extendr interface.

use extendr_api::*;

use crate::core::math::sparse::parse_compressed_sparse_format;
use crate::single_cell::sc_analysis::hdwgcna_meta_cells::MetaCellParams;
use crate::single_cell::sc_analysis::hotspot::HotSpotParams;
use crate::single_cell::sc_analysis::milo_r::MiloRParams;
use crate::single_cell::sc_analysis::scenic::{
    ExtraTreesConfig, GradientBoostingConfig, RandomForestConfig, RegressionLearner, ScenicParams,
};
use crate::single_cell::sc_analysis::seacells::SEACellsParams;
use crate::single_cell::sc_analysis::super_cells::SuperCellParams;
use crate::single_cell::sc_analysis::vision::SignatureGenes;
use crate::single_cell::sc_batch_correction::fast_mnn::FastMnnParams;
use crate::single_cell::sc_batch_correction::harmony::HarmonyParams;
use crate::single_cell::sc_data::data_io::MinCellQuality;
use crate::single_cell::sc_data::h5ad_multifile_io::H5adFileTask;
use crate::single_cell::sc_data::sc_synthetic_data::CellTypeConfig;
use crate::single_cell::sc_processing::doublet_detection::BoostParams;
use crate::single_cell::sc_processing::knn::KnnParams;
use crate::single_cell::sc_processing::scdblfinder::ScDblFinderParams;
use crate::single_cell::sc_processing::scrublet::ScrubletParams;

/////////////
// Helpers //
/////////////

/// Convert assignments to R-friendly list format with unassigned cell handling
///
/// Returns -1 for unassigned cells (R convention for missing/unassigned).
///
/// ### Params
///
/// * `assignments` - Vector where
///   `assignments[cell_id] = Some(metacell_id) or None`
/// * `n_cells` - Total number of cells
/// * `k` - Number of metacells
///
/// ### Returns
///
/// R List with -1 indicating unassigned cells
pub fn assignments_to_r_list(assignments: &[Option<usize>], n_cells: usize) -> List {
    let r_assignments: Vec<i32> = assignments
        .iter()
        .map(|&x| match x {
            Some(id) => (id + 1) as i32,
            None => -1,
        })
        .collect();

    let n_unassigned = assignments.iter().filter(|x| x.is_none()).count();

    let actual_k = assignments
        .iter()
        .filter_map(|&x| x)
        .max()
        .map(|x| x + 1)
        .unwrap_or(0);

    let mut metacells = vec![Vec::new(); actual_k];

    for (cell_id, &metacell_id) in assignments.iter().enumerate() {
        if let Some(id) = metacell_id {
            metacells[id].push((cell_id + 1) as i32);
        }
    }

    let unassigned: Vec<i32> = assignments
        .iter()
        .enumerate()
        .filter_map(|(cell_id, &x)| {
            if x.is_none() {
                Some((cell_id + 1) as i32)
            } else {
                None
            }
        })
        .collect();

    let metacells_list: List = metacells.into_iter().map(Robj::from).collect();

    list!(
        assignments = r_assignments,
        metacells = metacells_list,
        unassigned = unassigned,
        n_metacells = actual_k,
        n_cells = n_cells,
        n_unassigned = n_unassigned
    )
}

////////////////////
// Param wrappers //
////////////////////

//////////////
// CellType //
//////////////

impl CellTypeConfig {
    /// Generate the CellTypeConfig from an R list
    ///
    /// If values are not found, will use default values
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list containing the parameters. Should have the
    ///   elements `"marker_genes"`, `"marker_exp_range"`, `"markers_per_cell"`.
    ///
    /// ### Returns
    ///
    /// The `CellTypeConfig` based on the R list.
    pub fn from_r_list(r_list: List) -> Self {
        let map = r_list.into_hashmap();

        let marker_genes = map
            .get("marker_genes")
            .and_then(|v| v.as_integer_vector())
            .map(|v| v.iter().map(|x| *x as usize).collect())
            .unwrap_or_default();

        CellTypeConfig { marker_genes }
    }
}

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

        // ivf
        let n_list = params_list
            .get("n_list")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        let n_probe = params_list
            .get("n_probe")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

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
            n_list,
            n_probe,
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

///////////
// Boost //
///////////

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

///////////////////////
// ScDblFinderParams //
///////////////////////

impl ScDblFinderParams {
    /// Generate ScDblFinderParams from an R list.
    ///
    /// Values not found in the list fall back to the `Default` implementation.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with scDblFinder parameters.
    ///
    /// ### Returns
    ///
    /// `ScDblFinderParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Self {
        let knn_params = KnnParams::from_r_list(r_list.clone());
        let map = r_list.into_hashmap();
        let defaults = Self::default();

        Self {
            // Normalisation
            log_transform: map
                .get("log_transform")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.log_transform),
            mean_center: map
                .get("mean_center")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.mean_center),
            normalise_variance: map
                .get("normalise_variance")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.normalise_variance),
            target_size: map
                .get("target_size")
                .and_then(|v| v.as_real())
                .map(|x| x as f32),
            // HVG
            min_gene_var_pctl: map
                .get("min_gene_var_pctl")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.min_gene_var_pctl as f64) as f32,
            hvg_method: String::from(
                map.get("hvg_method")
                    .and_then(|v| v.as_str())
                    .unwrap_or("vst"),
            ),
            loess_span: map
                .get("loess_span")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.loess_span),
            clip_max: map
                .get("clip_max")
                .and_then(|v| v.as_real())
                .map(|x| x as f32),
            // PCA
            no_pcs: map
                .get("no_pcs")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.no_pcs as i32) as usize,
            random_svd: map
                .get("random_svd")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.random_svd),
            // Simulation
            doublet_ratio: map
                .get("doublet_ratio")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.doublet_ratio as f64) as f32,
            heterotypic_bias: map
                .get("heterotypic_bias")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.heterotypic_bias as f64) as f32,
            // Clustering
            cluster_resolution: map
                .get("cluster_resolution")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.cluster_resolution as f64)
                as f32,
            cluster_iters: map
                .get("cluster_iters")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.cluster_iters as i32) as usize,
            // kNN
            knn_params,
            // Iteration
            n_iterations: map
                .get("n_iterations")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.n_iterations as i32) as usize,
            // Classification
            n_trees: map
                .get("n_trees")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.n_trees as i32) as usize,
            max_depth: map
                .get("max_depth")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.max_depth as i32) as usize,
            learning_rate: map
                .get("learning_rate")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.learning_rate as f64) as f32,
            min_samples_leaf: map
                .get("min_samples_leaf")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.min_samples_leaf as i32) as usize,
            early_stop_window: map
                .get("early_stop_window")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.early_stop_window as i32)
                as usize,
            subsample_rate: map
                .get("subsample_rate")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.subsample_rate as f64) as f32,
            // Feature
            include_pcs: map
                .get("include_pcs")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.include_pcs as i32) as usize,
            // Thresholding
            manual_threshold: map
                .get("manual_threshold")
                .and_then(|v| v.as_real())
                .map(|x| x as f32),
            n_bins: map
                .get("n_bins")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.n_bins as i32) as usize,
            // Expected doublet rate
            dbr_per_1k: map
                .get("dbr_per_1k")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.dbr_per_1k as f64) as f32,
        }
    }
}

/////////////
// FastMNN //
/////////////

impl FastMnnParams {
    /// Generate the FastMnnParams from an R list
    ///
    /// Should values not be found within the List, the parameters will default
    /// to sensible defaults.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the fastMNN parameters.
    ///
    /// ### Return
    ///
    /// The `FastMnnParams` with all of the parameters.
    pub fn from_r_list(r_list: List) -> Self {
        let knn_params = KnnParams::from_r_list(r_list.clone());
        let fastmnn_list = r_list.into_hashmap();
        let ndist = fastmnn_list
            .get("ndist")
            .and_then(|v| v.as_real())
            .unwrap_or(3.0) as f32;
        let cos_norm = fastmnn_list
            .get("cos_norm")
            .and_then(|v| v.as_logical())
            .map(|rb| rb.is_true())
            .unwrap_or(true);
        let no_pcs = fastmnn_list
            .get("no_pcs")
            .and_then(|v| v.as_integer())
            .unwrap_or(30) as usize;
        let random_svd = fastmnn_list
            .get("random_svd")
            .and_then(|v| v.as_logical())
            .map(|rb| rb.is_true())
            .unwrap_or(true);
        Self {
            ndist,
            no_pcs,
            random_svd,
            cos_norm,
            knn_params,
        }
    }
}

///////////
// miloR //
///////////

impl MiloRParams {
    /// Generate MiloRParams from an R list
    ///
    /// Should values not be found within the List, the parameters will default
    /// to sensible defaults based on heuristics.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the MiloR parameters
    ///
    /// ### Returns
    ///
    /// The `MiloRParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Self {
        let knn_params = KnnParams::from_r_list(r_list.clone());

        let params_list = r_list.into_hashmap();

        let prop = params_list
            .get("prop")
            .and_then(|v| v.as_real())
            .unwrap_or(0.2);

        let k_refine = params_list
            .get("k_refine")
            .and_then(|v| v.as_integer())
            .unwrap_or(20) as usize;

        let index_type = std::string::String::from(
            params_list
                .get("index_type")
                .and_then(|v| v.as_str())
                .unwrap_or("annoy"),
        );

        let refinement_strategy = std::string::String::from(
            params_list
                .get("refinement_strategy")
                .and_then(|v| v.as_str())
                .unwrap_or("approximate"),
        );

        Self {
            prop,
            k_refine,
            refinement_strategy,
            index_type,
            knn_params,
        }
    }
}

//////////////
// SEACells //
//////////////

impl SEACellsParams {
    /// Generate SEACellsParams from an R list
    ///
    /// Should values not be found within the List, the parameters will default
    /// to sensible defaults based on the original SEACells implementation.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the SEACells parameters.
    ///
    /// ### Returns
    ///
    /// The `SEACellsParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Self {
        let knn_params = KnnParams::from_r_list(r_list.clone());

        let seacells_list = r_list.into_hashmap();

        let n_sea_cells = seacells_list
            .get("n_sea_cells")
            .and_then(|v| v.as_integer())
            .unwrap_or(0) as usize;

        let max_fw_iters = seacells_list
            .get("max_fw_iters")
            .and_then(|v| v.as_integer())
            .unwrap_or(50) as usize;

        // convergence_epsilon: algorithm converges when RSS change < epsilon * RSS(0)
        // Default: 1e-3 from Python implementation
        let convergence_epsilon = seacells_list
            .get("convergence_epsilon")
            .and_then(|v| v.as_real())
            .unwrap_or(1e-3) as f32;

        let max_iter = seacells_list
            .get("max_iter")
            .and_then(|v| v.as_integer())
            .unwrap_or(100) as usize;

        let min_iter = seacells_list
            .get("min_iter")
            .and_then(|v| v.as_integer())
            .unwrap_or(10) as usize;

        let greedy_threshold = seacells_list
            .get("greedy_threshold")
            .and_then(|v| v.as_integer())
            .unwrap_or(20000) as usize;

        let graph_building = seacells_list
            .get("graph_building")
            .and_then(|v| v.as_str())
            .unwrap_or("union")
            .to_string();

        let pruning = seacells_list
            .get("pruning")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let pruning_threshold: f32 = seacells_list
            .get("pruning_threshold")
            .and_then(|v| v.as_real())
            .unwrap_or(1e-7) as f32;

        Self {
            // seacell
            n_sea_cells,
            max_fw_iters,
            convergence_epsilon,
            max_iter,
            min_iter,
            greedy_threshold,
            graph_building,
            pruning,
            pruning_threshold,
            // knn
            knn_params,
        }
    }
}

///////////////
// MetaCells //
///////////////

impl MetaCellParams {
    /// Generate the MetaCellParams from an R list
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list with the parameters.
    ///
    /// ### Return
    ///
    /// The `MetaCellParams` structure.
    pub fn from_r_list(r_list: List) -> Self {
        let knn_params = KnnParams::from_r_list(r_list.clone());
        let meta_cell_params = r_list.into_hashmap();

        // meta cell
        let max_shared = meta_cell_params
            .get("max_shared")
            .and_then(|v| v.as_integer())
            .unwrap_or(15) as usize;
        let target_no_metacells = meta_cell_params
            .get("target_no_metacells")
            .and_then(|v| v.as_integer())
            .unwrap_or(1000) as usize;
        let max_iter = meta_cell_params
            .get("max_iter")
            .and_then(|v| v.as_integer())
            .unwrap_or(5000) as usize;

        Self {
            max_shared,
            target_no_metacells,
            max_iter,
            knn_params,
        }
    }
}

////////////////
// SuperCells //
////////////////

impl SuperCellParams {
    /// Generate the SuperCellParams from an R list
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list with the parameters
    ///
    /// ### Return
    ///
    /// The `SuperCellParams` structure
    pub fn from_r_list(r_list: List) -> Self {
        let knn_params = KnnParams::from_r_list(r_list.clone());

        let params = r_list.into_hashmap();

        // supercell
        let walk_length = params
            .get("walk_length")
            .and_then(|v| v.as_integer())
            .unwrap_or(3) as usize;

        let graining_factor = params
            .get("graining_factor")
            .and_then(|v| v.as_real())
            .unwrap_or(50.0);

        let linkage_dist = params
            .get("linkage_dist")
            .and_then(|v| v.as_str())
            .unwrap_or("average")
            .to_string();

        Self {
            walk_length,
            graining_factor,
            linkage_dist,
            knn_params,
        }
    }
}

////////////
// Vision //
////////////

impl SignatureGenes {
    /// Generate a SignatureGenes from an R list
    ///
    /// ### Params
    ///
    /// * `r_list` - An R list that is expected to have `"pos"` and `"neg"` with
    ///   0-index positions of the gene for this gene set.
    pub fn from_r_list(r_list: List) -> Self {
        let r_list = r_list.into_hashmap();

        let positive: Vec<usize> = r_list
            .get("pos")
            .and_then(|v| v.as_integer_vector())
            .unwrap_or_default()
            .iter()
            .map(|x| *x as usize)
            .collect();

        let negative: Vec<usize> = r_list
            .get("neg")
            .and_then(|v| v.as_integer_vector())
            .unwrap_or_default()
            .iter()
            .map(|x| *x as usize)
            .collect();

        Self { positive, negative }
    }
}

/////////////
// Hotspot //
/////////////

impl HotSpotParams {
    /// Generate HotSpotParams from an R list
    ///
    /// Should values not be found within the List, the parameters will default
    /// to sensible defaults based on heuristics.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the Boost parameters.
    ///
    /// ### Returns
    ///
    /// The `HotSpotParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Self {
        let knn_params = KnnParams::from_r_list(r_list.clone());

        let params_list = r_list.into_hashmap();

        // hotspot
        let model = std::string::String::from(
            params_list
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("normal"),
        );

        let normalise = params_list
            .get("normalise")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        Self {
            model,
            normalise,
            knn_params,
        }
    }
}

/////////////
// Harmony //
/////////////

impl HarmonyParams {
    /// Generate HarmonyParams from an R list.
    ///
    /// Should values not be found within the List, the parameters will default
    /// to the values defined in `HarmonyParams::default()`.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the Harmony parameters.
    ///
    /// ### Returns
    ///
    /// The `HarmonyParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Self {
        let defaults = Self::default();
        let params_list = r_list.into_hashmap();

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

        let block_size = params_list
            .get("block_size")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.block_size);

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

        Self {
            k,
            sigma,
            theta,
            lambda,
            block_size,
            max_iter_kmeans,
            max_iter_harmony,
            epsilon_kmeans,
            epsilon_harmony,
            window_size,
        }
    }
}

////////////////////
// SCENIC - Trees //
////////////////////

impl RandomForestConfig {
    /// Generate RandomForestConfig from an R list.
    ///
    /// Should values not be found within the List, the parameters will default
    /// to the values defined in `RandomForestConfig::default()`.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the RandomForest parameters.
    ///
    /// ### Returns
    ///
    /// The `RandomForestConfig` with all parameters set.
    pub fn from_r_list(r_list: List) -> Self {
        let defaults = Self::default();
        let params_list = r_list.into_hashmap();
        let n_trees = params_list
            .get("n_trees")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.n_trees);
        let min_samples_leaf = params_list
            .get("min_samples_leaf")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.min_samples_leaf);
        let n_features_split = params_list
            .get("n_features_split")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.n_features_split);
        let subsample_rate = params_list
            .get("subsample_rate")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.subsample_rate);
        let bootstrap = params_list
            .get("bootstrap")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.bootstrap);
        let max_depth = params_list
            .get("max_depth")
            .and_then(|v| v.as_integer())
            .map(|v| Some(v as usize))
            .unwrap_or(defaults.max_depth);
        let subsample_frac = params_list
            .get("subsample_frac")
            .and_then(|v| v.as_real())
            .map(|v| Some(v as f32))
            .unwrap_or(defaults.subsample_frac);
        Self {
            n_trees,
            min_samples_leaf,
            n_features_split,
            subsample_rate,
            bootstrap,
            max_depth,
            subsample_frac,
        }
    }
}

impl ExtraTreesConfig {
    /// Generate ExtraTreesConfig from an R list.
    ///
    /// Should values not be found within the List, the parameters will default
    /// to the values defined in `ExtraTreesConfig::default()`.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the ExtraTrees parameters.
    ///
    /// ### Returns
    ///
    /// The `ExtraTreesConfig` with all parameters set.
    pub fn from_r_list(r_list: List) -> Self {
        let defaults = Self::default();
        let params_list = r_list.into_hashmap();
        let n_trees = params_list
            .get("n_trees")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.n_trees);
        let min_samples_leaf = params_list
            .get("min_samples_leaf")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.min_samples_leaf);
        let n_features_split = params_list
            .get("n_features_split")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.n_features_split);
        let n_thresholds = params_list
            .get("n_thresholds")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.n_thresholds);
        let max_depth = params_list
            .get("max_depth")
            .and_then(|v| v.as_integer())
            .map(|v| Some(v as usize))
            .unwrap_or(defaults.max_depth);
        let subsample_frac = params_list
            .get("subsample_frac")
            .and_then(|v| v.as_real())
            .map(|v| Some(v as f32))
            .unwrap_or(defaults.subsample_frac);
        Self {
            n_trees,
            min_samples_leaf,
            n_features_split,
            n_thresholds,
            max_depth,
            subsample_frac,
        }
    }
}

impl GradientBoostingConfig {
    /// Generate GradientBoostingConfig from an R list.
    ///
    /// Should values not be found within the List, the parameters will default
    /// to the values defined in `GradientBoostingConfig::default()`.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the GradientBoosting parameters.
    ///
    /// ### Returns
    ///
    /// The `GradientBoostingConfig` with all parameters set.
    pub fn from_r_list(r_list: List) -> Self {
        let defaults = Self::default();
        let params_list = r_list.into_hashmap();
        let n_trees_max = params_list
            .get("n_trees_max")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.n_trees_max);
        let learning_rate = params_list
            .get("learning_rate")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.learning_rate);
        let max_depth = params_list
            .get("max_depth")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.max_depth);
        let min_samples_leaf = params_list
            .get("min_samples_leaf")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.min_samples_leaf);
        let early_stop_window = params_list
            .get("early_stop_window")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.early_stop_window);
        let subsample_rate = params_list
            .get("subsample_rate")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.subsample_rate);
        let n_features_split = params_list
            .get("n_features_split")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.n_features_split);
        Self {
            n_trees_max,
            learning_rate,
            max_depth,
            min_samples_leaf,
            early_stop_window,
            subsample_rate,
            n_features_split,
        }
    }
}

/////////////////////
// SCENIC - Params //
/////////////////////

impl ScenicParams {
    /// Generate `ScenicParams` from an R list.
    ///
    /// The list is expected to contain fields for all top-level SCENIC
    /// parameters plus the fields required by the chosen regression learner.
    /// Tree configuration parameters are read from the same flat list as the
    /// top-level parameters, following the same convention as `KnnParams` in
    /// other parameter structs.
    ///
    /// The `learner_type` field selects the regression learner:
    /// `"extratrees"` maps to `ExtraTreesConfig`, anything else (including the
    /// default `"randomforest"`) maps to `RandomForestConfig`. Learner
    /// parameters are then parsed from the same list via the respective
    /// `from_r_list` implementation.
    ///
    /// Should values not be found within the list, parameters will default to
    /// the values defined in `ScenicParams::default()`.
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list with the SCENIC parameters.
    ///
    /// ### Returns
    ///
    /// The `ScenicParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Self {
        let defaults = Self::default();
        let params = r_list.clone().into_hashmap();

        let min_counts = params
            .get("min_counts")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.min_counts);

        let min_cells = params
            .get("min_cells")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.min_cells);

        let gene_batch_strategy = params
            .get("gene_batch_strategy")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or(defaults.gene_batch_strategy);

        let gene_batch_size = params
            .get("gene_batch_size")
            .and_then(|v| v.as_integer())
            .map(|v| Some(v as usize))
            .unwrap_or(defaults.gene_batch_size);

        let n_pcs = params
            .get("n_pcs")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.n_pcs);

        let n_subsample = params
            .get("n_subsample")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.n_subsample);

        let learner_type = params
            .get("learner_type")
            .and_then(|v| v.as_str())
            .unwrap_or("randomforest");

        let regression_learner = match learner_type.to_lowercase().as_str() {
            "extratrees" => RegressionLearner::ExtraTrees(ExtraTreesConfig::from_r_list(r_list)),
            "grnboost2" => {
                RegressionLearner::GradientBoosting(GradientBoostingConfig::from_r_list(r_list))
            }
            _ => RegressionLearner::RandomForest(RandomForestConfig::from_r_list(r_list)),
        };

        Self {
            min_counts,
            min_cells,
            regression_learner,
            gene_batch_strategy,
            gene_batch_size,
            n_pcs,
            n_subsample,
        }
    }
}

//////////////////
// H5adFileTask //
//////////////////

impl H5adFileTask {
    /// Generate an H5FileTask from an R list
    ///
    /// Expects: exp_id, h5_path, cs_type, no_cells, no_genes,
    /// gene_local_to_universe (integer vector, NA for unmapped genes,
    /// 0-indexed).
    pub fn from_r_list(r_list: List) -> Self {
        let map = r_list.into_hashmap();

        let exp_id = map
            .get("exp_id")
            .and_then(|v| v.as_str())
            .expect("exp_id missing or not a string")
            .to_string();

        let h5_path = map
            .get("h5_path")
            .and_then(|v| v.as_str())
            .expect("h5_path missing or not a string")
            .to_string();

        let cs_type_str = map
            .get("cs_type")
            .and_then(|v| v.as_str())
            .expect("cs_type missing or not a string");
        let cs_type =
            parse_compressed_sparse_format(cs_type_str).expect("cs_type must be 'csr' or 'csc'");

        let no_cells = map
            .get("no_cells")
            .and_then(|v| v.as_integer())
            .expect("no_cells missing") as usize;

        let no_genes = map
            .get("no_genes")
            .and_then(|v| v.as_integer())
            .expect("no_genes missing") as usize;

        let mapping_robj = map
            .get("gene_local_to_universe")
            .expect("gene_local_to_universe missing");
        let mapping_raw: Vec<i32> = mapping_robj
            .as_integer_slice()
            .expect("gene_local_to_universe must be integer vector")
            .to_vec();

        // R NA_integer_ is i32::MIN
        let gene_local_to_universe: Vec<Option<usize>> = mapping_raw
            .into_iter()
            .map(|v| {
                if v == i32::MIN {
                    None
                } else {
                    Some(v as usize)
                }
            })
            .collect();

        Self {
            exp_id,
            h5_path,
            cs_type,
            no_cells,
            no_genes,
            gene_local_to_universe,
        }
    }
}
