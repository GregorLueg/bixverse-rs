//! Contains R-specific functions for single-cell data processing that need
//! the extendr interface.

use edge_rs::sc::nebula::NebulaParams;
use edge_rs::sc::test::ScTested;
use extendr_api::*;
use std::collections::HashMap;

use crate::ml::clustering::k_means::KMeansParamsWrappers;
use crate::ml::gp::LandmarkGpParams;
use crate::prelude::{BixverseFloat, LanczosParams, VecConvert, VecFloatConvert};
use crate::single_cell::mc_generation::{
    hdwgcna_meta_cells::BootstrappedMetaCellParams, metacells2::params::*,
    seacells::SEACellsParams, super_cells::SuperCellParams,
};
use crate::single_cell::sc_analysis::{
    dge_pathway_scores::{AucellParams, parse_auc_type},
    dialogue::{DialogueParams, HlmParams, PmdParams, RefineParams, parse_averaging},
    fast_clusters::FastLouvainParams,
    hotspot::{HotSpotGraphParams, HotSpotParams},
    meld::{MeldParams, parse_lap_type, parse_meld_filter},
    milo_r::MiloRParams,
    nebula::{NebulaScParams, parse_nebula_method},
    nichenet::ligand_regulatory_potential::LigandTargetParams,
    regulon_binarise::BinariseParams,
    scenic::{
        ExtraTreesConfig, GradientBoostingConfig, RandomForestConfig, RegressionLearner,
        ScenicParams,
    },
    vision::SignatureGenes,
};
use crate::single_cell::sc_data::h5ad_io::parse_h5ad_format;
use crate::single_cell::sc_processing::magic::{MagicLayer, MagicParams};
use crate::single_cell::sc_trajectory::gene_trends::{
    BranchSelectionParams, BranchWeighting, GeneTrendsParams,
};
use crate::single_cell::sc_trajectory::palantir::PalantirParams;
use crate::utils::r_rust_interface::{r_list_count, r_list_count_allow_zero, r_list_to_map};

use crate::single_cell::sc_annotation::{
    sc_type::{CellTypeMarkers, ScTypeCellParams, SctypeRes, parse_score_calibration},
    symphony::SymphonyMapParams,
};
use crate::single_cell::sc_batch_correction::{
    bbknn::BbknnParams, fast_mnn::FastMnnParams, harmony::HarmonyParams,
    harmony_v2::HarmonyParamsV2, seurat_cca::SeuratCcaParams, seurat_rpca::SeuratRpcaParams,
};
use crate::single_cell::sc_data::h5ad_io::RawDataSlot;
use crate::single_cell::sc_data::{
    bin_merge_io::BinMergeTask, data_io::MinCellQuality, h5_10x_io::parse_tenx_version,
    h5_10x_multifile_io::TenxFileTask, h5ad_io::parse_raw_slot, h5ad_multifile_io::H5adFileTask,
    mtx_multifile_io::MtxFileTask, sc_synthetic_data::CellTypeConfig,
};
use crate::single_cell::sc_processing::{
    doublet_detection::BoostParams, knn::KnnParams, pca::SingleCellPcaParams,
    scdblfinder::ScDblFinderParams, scrublet::ScrubletParams, utils_doublets::ScDblSimParams,
};

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
///
/// ### Returns
///
/// R List with the slots `assignments`, `metacells`, `unassigned`,
/// `n_metacells`, `n_cells` and `n_unassigned`, with -1 marking unassigned
/// cells. `n_metacells` is derived as `max(id) + 1` over the assignments, so
/// trailing metacells that ended up empty are dropped rather than reported at
/// size zero.
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

/// Build the R-facing assignment list from metacell membership lists.
///
/// Cells may appear in multiple metacells (overlap is allowed). The per-cell
/// `assignments` field is therefore a list of integer vectors, one per cell,
/// each containing the 1-indexed metacell IDs that cell belongs to. Cells
/// in no metacell get an empty vector and appear in `unassigned`. This
/// helper is needed for the bootstrapped approach of meta cell generation
///
/// ### Params
///
/// * `metacell` - A slice of the cell indices per metacell
/// * `n_cells` - Total number of cells
///
/// ### Returns
///
/// R list in a similar structure as [assignments_to_r_list] with a difference
/// in the assigmnents. These are now a list, not an integer vector!
pub fn metacells_to_r_list(metacells: &[Vec<usize>], n_cells: usize) -> List {
    let mut cell_assignments: Vec<Vec<i32>> = vec![Vec::new(); n_cells];
    for (mc_id, cells) in metacells.iter().enumerate() {
        for &cell_id in cells {
            cell_assignments[cell_id].push((mc_id + 1) as i32);
        }
    }

    let unassigned: Vec<i32> = cell_assignments
        .iter()
        .enumerate()
        .filter_map(|(cell_id, mcs)| mcs.is_empty().then_some((cell_id + 1) as i32))
        .collect();

    let n_unassigned = unassigned.len();
    let n_metacells = metacells.len();

    let metacells_list: List = metacells
        .iter()
        .map(|cells| {
            let r_cells: Vec<i32> = cells.iter().map(|&c| (c + 1) as i32).collect();
            Robj::from(r_cells)
        })
        .collect();

    let assignments_list: List = cell_assignments.into_iter().map(Robj::from).collect();

    list!(
        assignments = assignments_list,
        metacells = metacells_list,
        unassigned = unassigned,
        n_metacells = n_metacells,
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
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let map: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let marker_genes = map
            .get("marker_genes")
            .and_then(|v| v.as_integer_vector())
            .map(|v| v.iter().map(|x| *x as usize).collect())
            .unwrap_or_default();

        Ok(CellTypeConfig { marker_genes })
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
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let min_qc: HashMap<&str, Robj> = r_list_to_map(r_list)?;

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

        Ok(MinCellQuality {
            min_unique_genes,
            min_lib_size,
            min_cells,
            target_size,
        })
    }
}

///////////////
// KnnParams //
///////////////

impl KnnParams {
    /// Generate KnnParams from an R list
    ///
    /// Should values not be found within the List, the parameters will default
    /// to sensible defaults based on heuristics. Every count is read through
    /// [r_list_count], so `list(k = 50)` works as well as `list(k = 50L)` and a
    /// nonsensical value errors instead of silently reverting to the default.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the kNN parameters.
    ///
    /// ### Returns
    ///
    /// The `KnnParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;

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

        // this allows zeros through deliberately for auto-detection in the
        // doublet detection methods
        let k = r_list_count_allow_zero(&params_list, "k")?.unwrap_or(15);

        // annoy
        let n_tree = r_list_count(&params_list, "n_trees")?.unwrap_or(50);

        let search_budget = r_list_count(&params_list, "search_budget")?;

        // nn descent
        let diversify_prob = params_list
            .get("diversify_prob")
            .and_then(|v| v.as_real())
            .unwrap_or(0.0) as f32;

        let delta = params_list
            .get("delta")
            .and_then(|v| v.as_real())
            .unwrap_or(0.001) as f32;

        let ef_budget = r_list_count(&params_list, "ef_budget")?;
        let extract_knn = params_list
            .get("extract_knn")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // hnsw
        let m = r_list_count(&params_list, "m")?.unwrap_or(16);

        let ef_construction = r_list_count(&params_list, "ef_construction")?.unwrap_or(200);

        let ef_search = r_list_count(&params_list, "ef_search")?.unwrap_or(100);

        // ivf
        let n_list = r_list_count(&params_list, "n_list")?;

        let n_probe = r_list_count(&params_list, "n_probe")?;

        Ok(Self {
            knn_method,
            ann_dist,
            k,
            n_tree,
            search_budget,
            ef_budget,
            extract_knn,
            diversify_prob,
            delta,
            m,
            ef_construction,
            ef_search,
            n_list,
            n_probe,
        })
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
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let knn_params = KnnParams::from_r_list(r_list.clone())?;

        let scrublet_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;

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

        let n_bins = scrublet_list
            .get("n_bins")
            .and_then(|v| v.as_integer())
            .unwrap_or(20) as usize;

        let binning_strategy = std::string::String::from(
            scrublet_list
                .get("binning_strategy")
                .and_then(|v| v.as_str())
                .unwrap_or("equal_width"),
        );

        // PCA parameters
        let no_pcs = scrublet_list
            .get("no_pcs")
            .and_then(|v| v.as_integer())
            .unwrap_or(30) as usize;

        let random_svd = scrublet_list
            .get("random_svd")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

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

        // Doublet calling parameters. `params_scrublet()` emits
        // `n_bins_histogram`; `n_bins_hist` is the Rust field name and stays
        // accepted for hand-built lists. Reading only the latter, as this did,
        // meant the user's bin count never reached the Otsu histogram.
        let n_bins_hist = scrublet_list
            .get("n_bins_histogram")
            .or_else(|| scrublet_list.get("n_bins_hist"))
            .and_then(|v| v.as_integer())
            .unwrap_or(50) as usize;

        let manual_threshold = scrublet_list
            .get("manual_threshold")
            .and_then(|v| v.as_real())
            .map(|x| x as f32);

        Ok(Self {
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
            n_bins,
            binning_strategy,
            // pca
            no_pcs,
            random_svd,
            // doublet simulation/detection
            sim_doublet_ratio,
            expected_doublet_rate,
            stdev_doublet_rate,
            n_bins_hist,
            manual_threshold,
            // knn
            knn_params,
        })
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
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let knn_params = KnnParams::from_r_list(r_list.clone())?;
        let fast_cluster_params = FastLouvainParams::from_r_list(r_list.clone())?;

        let params_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;

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

        let n_bins = params_list
            .get("n_bins")
            .and_then(|v| v.as_integer())
            .unwrap_or(20) as usize;

        let binning_strategy = std::string::String::from(
            params_list
                .get("binning_strategy")
                .and_then(|v| v.as_str())
                .unwrap_or("equal_width"),
        );

        // pca
        let no_pcs = params_list
            .get("no_pcs")
            .and_then(|v| v.as_integer())
            .unwrap_or(30) as usize;

        let random_svd = params_list
            .get("random_svd")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

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

        let p_thresh = params_list
            .get("p_thresh")
            .and_then(|v| v.as_real())
            .unwrap_or(1e-7) as f32;

        let voter_thresh = params_list
            .get("voter_thresh")
            .and_then(|v| v.as_real())
            .unwrap_or(0.9) as f32;

        let fast_cluster = params_list
            .get("fast_cluster")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let km_type = std::string::String::from(
            params_list
                .get("km_type")
                .and_then(|v| v.as_str())
                .unwrap_or("minibatch"),
        );

        Ok(Self {
            // processing
            log_transform,
            mean_center,
            normalise_variance,
            target_size,
            // hvg
            min_gene_var_pctl,
            hvg_method,
            loess_span,
            clip_max,
            n_bins,
            binning_strategy,
            // pca
            no_pcs,
            random_svd,
            // boosted
            boost_rate,
            replace,
            resolution,
            louvain_iters,
            n_iters,
            p_thresh,
            voter_thresh,
            // knn
            knn_params,
            // fast cluster related stuff
            fast_cluster,
            fast_cluster_params,
            km_type,
        })
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
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let knn_params = KnnParams::from_r_list(r_list.clone())?;
        let fast_cluster_params = FastLouvainParams::from_r_list(r_list.clone())?;
        let map: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = Self::default();

        let km_type = std::string::String::from(
            map.get("km_type")
                .and_then(|v| v.as_str())
                .unwrap_or("minibatch"),
        );

        Ok(Self {
            // Preprocessing
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
            n_genes: map
                .get("n_genes")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.n_genes as i32) as usize,
            // Simulation
            doublet_ratio: map
                .get("doublet_ratio")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.doublet_ratio as f64) as f32,
            heterotypic_bias: map
                .get("heterotypic_bias")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.heterotypic_bias as f64) as f32,
            sim_params: ScDblSimParams::default(),
            // PCA
            no_pcs: map
                .get("no_pcs")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.no_pcs as i32) as usize,
            random_svd: map
                .get("random_svd")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.random_svd),
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
            fast_cluster: map
                .get("fast_cluster")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.fast_cluster),
            km_type,
            fast_cluster_params,
            // kNN
            knn_params,
            // Iteration
            n_iterations: map
                .get("n_iterations")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.n_iterations as i32) as usize,
            // Classification
            gbm_n_trees: map
                .get("n_trees")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.gbm_n_trees as i32) as usize,
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
            subsample_rate: map
                .get("subsample_rate")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.subsample_rate as f64) as f32,
            cv_folds: map
                .get("cv_folds")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.cv_folds as i32) as usize,
            cv_early_stop: map
                .get("cv_early_stop")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.cv_early_stop as i32) as usize,
            se_fraction: map
                .get("se_fraction")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.se_fraction as f64) as f32,
            // Feature engineering
            include_pcs: map
                .get("include_pcs")
                .and_then(|v| v.as_integer())
                .unwrap_or(defaults.include_pcs as i32) as usize,
            return_features: map
                .get("return_features")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.return_features),
            cxds_genes: map
                .get("cxds_genes")
                .and_then(|v| v.as_integer())
                .map(|x| x as usize),
            // Expected doublet rate
            expected_doublet_rate: map
                .get("expected_doublet_rate")
                .and_then(|v| v.as_real())
                .map(|x| x as f32),
            // Thresholding
            manual_threshold: map
                .get("manual_threshold")
                .and_then(|v| v.as_real())
                .map(|x| x as f32),
        })
    }
}

///////////////////////
// FastLouvainParams //
///////////////////////

impl FastLouvainParams<f32> {
    /// Generate the [FastLouvainParams] from an R list.
    ///
    /// Should values not be found within the List, the parameters will default
    /// to sensible defaults. The nested kNN and k-means blocks are parsed from
    /// the same flat list via `KnnParams::from_r_list` and
    /// `KMeansParamsWrappers::from_r_list`.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the fast Louvain parameters.
    ///
    /// ### Return
    ///
    /// The [FastLouvainParams] with all of the parameters.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let knn_params = KnnParams::from_r_list(r_list.clone())?;
        let kmeans_params = KMeansParamsWrappers::from_r_list(r_list.clone())?;
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults: FastLouvainParams<f32> = Self::default();

        // knn
        let same_weight = params
            .get("same_weight")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.same_weight);

        // k means
        let n_centroids = params
            .get("n_centroids")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.n_centroids);

        let batch_size = params
            .get("batch_size")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.batch_size);

        let drift_threshold: f32 = params
            .get("drift_threshold")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.drift_threshold);

        let lr_alpha: f32 = params
            .get("lr_alpha")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.lr_alpha);

        // louvain
        let louvain_iters = params
            .get("louvain_iters")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.louvain_iters);
        let multi_level_louvain = params
            .get("multi_level_louvain")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.multi_level_louvain);

        // snn
        let full_snn = params
            .get("full_snn")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.full_snn);

        let pruning = params
            .get("pruning")
            .and_then(|v| v.as_real())
            .map(|x| x as f32);

        let snn_similarity = std::string::String::from(
            params
                .get("snn_similarity")
                .and_then(|v| v.as_str())
                .unwrap_or("jaccard"),
        );

        Ok(Self {
            n_centroids,
            kmeans_params,
            batch_size,
            drift_threshold,
            lr_alpha,
            louvain_iters,
            knn_params,
            full_snn,
            pruning,
            snn_similarity,
            multi_level_louvain,
            same_weight,
        })
    }
}

/////////
// PCA //
/////////

impl SingleCellPcaParams {
    /// Generate the [SingleCellPcaParams] from an R list.
    ///
    /// Should values not be found within the List, the parameters will default
    /// to sensible defaults.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the BBKNN parameters.
    ///
    /// ### Return
    ///
    /// The [SingleCellPcaParams] with all of the parameters.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = SingleCellPcaParams::default();

        let mean_center = params
            .get("mean_center")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.mean_center);

        let normalise_variance = params
            .get("normalise_variance")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.normalise_variance);

        let randomised = params
            .get("randomised")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.randomised);

        let clr = params
            .get("clr")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.clr);

        let size_factor = params
            .get("size_factor")
            .and_then(|v| v.as_real())
            .unwrap_or(1e4) as f32;

        Ok(Self {
            mean_center,
            normalise_variance,
            randomised,
            clr,
            size_factor,
        })
    }
}

///////////
// BBKNN //
///////////

impl BbknnParams {
    /// Generate the BbknnParams from a R list
    ///
    /// Should values not be found within the List, the parameters will default
    /// to sensible defaults.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the BBKNN parameters.
    ///
    /// ### Return
    ///
    /// The `BbknnParams` with all of the parameters.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let knn_params = KnnParams::from_r_list(r_list.clone())?;

        let bbknn_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let neighbours_within_batch = bbknn_list
            .get("neighbours_within_batch")
            .and_then(|v| v.as_integer())
            .unwrap_or(3) as usize;

        let set_op_mix_ratio = bbknn_list
            .get("set_op_mix_ratio")
            .and_then(|v| v.as_real())
            .unwrap_or(1.0) as f32;

        let local_connectivity = bbknn_list
            .get("local_connectivity")
            .and_then(|v| v.as_real())
            .unwrap_or(1.0) as f32;

        let trim = bbknn_list
            .get("trim")
            .and_then(|v| v.as_integer())
            .unwrap_or(10 * neighbours_within_batch as i32) as usize;

        Ok(Self {
            neighbours_within_batch,
            set_op_mix_ratio,
            local_connectivity,
            trim: Some(trim),
            knn_params,
        })
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
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let knn_params = KnnParams::from_r_list(r_list.clone())?;
        let pca_params = SingleCellPcaParams::from_r_list(r_list.clone())?;
        let fastmnn_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;

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
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let sparse_svd = fastmnn_list
            .get("sparse_svd")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        Ok(Self {
            ndist,
            no_pcs,
            random_svd,
            sparse_svd,
            cos_norm,
            knn_params,
            pca_params,
        })
    }
}

////////////////
// Seurat CCA //
////////////////

impl SeuratCcaParams {
    /// Generate the SeuratCcaParams from an R list
    ///
    /// Should values not be found within the list, the parameters will
    /// default to the values in `SeuratCcaParams::default()` (Seurat v3
    /// defaults).
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the Seurat CCA parameters, plus nested
    ///   `knn_params` and PCA fields consumed by [KnnParams::from_r_list]
    ///   and [SingleCellPcaParams::from_r_list].
    ///
    /// ### Returns
    ///
    /// The `SeuratCcaParams` with all of the parameters.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let knn_params = KnnParams::from_r_list(r_list.clone())?;
        let pca_params = SingleCellPcaParams::from_r_list(r_list.clone())?;
        let cca_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let num_cc = cca_list
            .get("num_cc")
            .and_then(|v| v.as_integer())
            .unwrap_or(30) as usize;
        let dims = cca_list
            .get("dims")
            .and_then(|v| v.as_integer())
            .unwrap_or(30) as usize;
        let k_anchor = cca_list
            .get("k_anchor")
            .and_then(|v| v.as_integer())
            .unwrap_or(5) as usize;
        let k_filter = cca_list
            .get("k_filter")
            .and_then(|v| v.as_integer())
            .unwrap_or(200) as usize;
        let k_score = cca_list
            .get("k_score")
            .and_then(|v| v.as_integer())
            .unwrap_or(30) as usize;
        let k_weight = cca_list
            .get("k_weight")
            .and_then(|v| v.as_integer())
            .unwrap_or(100) as usize;
        let n_top_features = cca_list
            .get("n_top_features")
            .and_then(|v| v.as_integer())
            .unwrap_or(200) as usize;
        let l2_norm = cca_list
            .get("l2_norm")
            .and_then(|v| v.as_logical())
            .map(|rb| rb.is_true())
            .unwrap_or(true);
        let sd = cca_list.get("sd").and_then(|v| v.as_real()).unwrap_or(1.0) as f32;

        Ok(Self {
            num_cc,
            dims,
            k_anchor,
            k_filter,
            k_score,
            k_weight,
            n_top_features,
            l2_norm,
            sd,
            knn_params,
            pca_params,
        })
    }
}

/////////////////
// Seurat rPCA //
/////////////////

impl SeuratRpcaParams {
    /// Generate the SeuratRpcaParams from an R list
    ///
    /// Should values not be found within the list, the parameters will
    /// default to the values in `SeuratRpcaParams::default()`. Note that
    /// rPCA does not use `num_cc`, `k_filter`, or `n_top_features` (the
    /// gene-space FilterAnchors step is CCA-only in Seurat).
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the Seurat rPCA parameters, plus nested
    ///   `knn_params` and PCA fields consumed by [KnnParams::from_r_list]
    ///   and [SingleCellPcaParams::from_r_list].
    ///
    /// ### Returns
    ///
    /// The `SeuratRpcaParams` with all of the parameters.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let knn_params = KnnParams::from_r_list(r_list.clone())?;
        let pca_params = SingleCellPcaParams::from_r_list(r_list.clone())?;
        let rpca_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let dims = rpca_list
            .get("dims")
            .and_then(|v| v.as_integer())
            .unwrap_or(30) as usize;
        let k_anchor = rpca_list
            .get("k_anchor")
            .and_then(|v| v.as_integer())
            .unwrap_or(5) as usize;
        let k_score = rpca_list
            .get("k_score")
            .and_then(|v| v.as_integer())
            .unwrap_or(30) as usize;
        let k_weight = rpca_list
            .get("k_weight")
            .and_then(|v| v.as_integer())
            .unwrap_or(100) as usize;
        let l2_norm = rpca_list
            .get("l2_norm")
            .and_then(|v| v.as_logical())
            .map(|rb| rb.is_true())
            .unwrap_or(true);
        let sd = rpca_list.get("sd").and_then(|v| v.as_real()).unwrap_or(1.0) as f32;

        Ok(Self {
            dims,
            k_anchor,
            k_score,
            k_weight,
            l2_norm,
            sd,
            knn_params,
            pca_params,
        })
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
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let knn_params = KnnParams::from_r_list(r_list.clone())?;

        let params_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;

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

        Ok(Self {
            prop,
            k_refine,
            refinement_strategy,
            index_type,
            knn_params,
        })
    }
}

//////////////
// SEACells //
//////////////

impl SEACellsParams {
    /// Generate SEACellsParams from an R list
    ///
    /// Any element missing from the list falls back to the matching field of
    /// [SEACellsParams::default], so the two sides cannot drift apart.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the SEACells parameters.
    ///
    /// ### Returns
    ///
    /// The `SEACellsParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let defaults = Self::default();
        let knn_params = KnnParams::from_r_list(r_list.clone())?;

        let seacells_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let lanczos_params = LanczosParams::from_r_list(&seacells_list)?;

        let n_sea_cells = seacells_list
            .get("n_sea_cells")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.n_sea_cells);

        let max_fw_iters = seacells_list
            .get("max_fw_iters")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.max_fw_iters);

        // convergence_epsilon: algorithm converges when RSS change < epsilon * RSS(0)
        let convergence_epsilon = seacells_list
            .get("convergence_epsilon")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.convergence_epsilon);

        let max_iter = seacells_list
            .get("max_iter")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.max_iter);

        let min_iter = seacells_list
            .get("min_iter")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.min_iter);

        let greedy_threshold = seacells_list
            .get("greedy_threshold")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.greedy_threshold);

        let graph_building = seacells_list
            .get("graph_building")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or(defaults.graph_building);

        let pruning = seacells_list
            .get("pruning")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.pruning);

        let pruning_threshold = seacells_list
            .get("pruning_threshold")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.pruning_threshold);

        let n_landmarks = seacells_list
            .get("n_landmarks")
            .and_then(|v| v.as_integer())
            .map(|x| x as usize);

        Ok(Self {
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
            n_landmarks,
            // knn
            knn_params,
            // eigensolver
            lanczos_params,
        })
    }
}

///////////////
// MetaCells //
///////////////

impl BootstrappedMetaCellParams {
    /// Generate the MetaCellParams from an R list
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list with the parameters.
    ///
    /// ### Return
    ///
    /// The `MetaCellParams` structure.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let knn_params = KnnParams::from_r_list(r_list.clone())?;
        let meta_cell_params: HashMap<&str, Robj> = r_list_to_map(r_list)?;

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

        Ok(Self {
            max_shared,
            target_no_metacells,
            max_iter,
            knn_params,
        })
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
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let knn_params = KnnParams::from_r_list(r_list.clone())?;

        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        // supercell
        let walk_length = params
            .get("walk_length")
            .and_then(|v| v.as_integer())
            .unwrap_or(3) as usize;

        let graining_factor = params
            .get("graining_factor")
            .and_then(|v| v.as_real())
            .unwrap_or(50.0);

        let use_kernel = params
            .get("use_kernel")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let k_ith = params
            .get("k_ith")
            .and_then(|v| v.as_integer())
            .map(|x| x as usize);
        let max_support = params
            .get("max_support")
            .and_then(|v| v.as_integer())
            .map(|x| x as usize);

        Ok(Self {
            walk_length,
            graining_factor,
            knn_params,
            use_kernel,
            k_ith,
            max_support,
        })
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
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let r_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;

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

        Ok(Self { positive, negative })
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
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let knn_params = KnnParams::from_r_list(r_list.clone())?;

        let params_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;

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

        // both default to what `Hotspot.create_knn_graph` ships upstream
        let defaults = HotSpotGraphParams::default();
        let weighted_graph = params_list
            .get("weighted_graph")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.weighted_graph);

        let neighborhood_factor = params_list
            .get("neighborhood_factor")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .filter(|v| *v > 0.0)
            .unwrap_or(defaults.neighborhood_factor);

        Ok(Self {
            model,
            normalise,
            knn_params,
            graph_params: HotSpotGraphParams::new(weighted_graph, neighborhood_factor),
        })
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
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let defaults = Self::default();
        let kmeans_params = KMeansParamsWrappers::from_r_list(r_list.clone())?;
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

        Ok(Self {
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
            kmeans_params,
        })
    }
}

//////////////////
// Harmony (v2) //
//////////////////

impl HarmonyParamsV2 {
    /// Generate HarmonyParamsV2 from an R list.
    ///
    /// Should values not be found within the List, the parameters will default
    /// to the values defined in `HarmonyParamsV2::default()`.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the Harmony parameters.
    ///
    /// ### Returns
    ///
    /// The `HarmonyParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let defaults = Self::default();
        let kmeans_params = KMeansParamsWrappers::from_r_list(r_list.clone())?;
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

        Ok(Self {
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
            alpha,
            tau,
            batch_proportion_cutoff,
            use_dynamic_lambda,
            kmeans_params,
        })
    }
}

//////////////
// Symphomy //
//////////////

/////////////////////
// Symphony params //
/////////////////////

impl SymphonyMapParams {
    /// Generate [SymphonyMapParams] from an R list.
    ///
    /// Should values not be found within the list, parameters will default
    /// to sensible defaults based on Symphony R conventions.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the Symphony mapping parameters.
    ///
    /// ### Returns
    ///
    /// The [SymphonyMapParams] with all parameters set.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = SymphonyMapParams::default();

        let sigma = params_list
            .get("sigma")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.sigma);
        let lambda = params_list
            .get("lambda")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.lambda);

        Ok(Self { sigma, lambda })
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
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let defaults = Self::default();
        let params_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;
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

        Ok(Self {
            n_trees,
            min_samples_leaf,
            n_features_split,
            subsample_rate,
            bootstrap,
            max_depth,
            subsample_frac,
        })
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
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let defaults = Self::default();
        let params_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;
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

        Ok(Self {
            n_trees,
            min_samples_leaf,
            n_features_split,
            n_thresholds,
            max_depth,
            subsample_frac,
        })
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
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let defaults = Self::default();
        let params_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;
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

        Ok(Self {
            n_trees_max,
            learning_rate,
            max_depth,
            min_samples_leaf,
            early_stop_window,
            subsample_rate,
            n_features_split,
        })
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
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let defaults = Self::default();
        let params: HashMap<&str, Robj> = r_list_to_map(r_list.clone())?;

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
            "extratrees" => {
                RegressionLearner::ExtraTrees(ExtraTreesConfig::from_r_list(r_list.clone())?)
            }
            "grnboost2" => RegressionLearner::GradientBoosting(
                GradientBoostingConfig::from_r_list(r_list.clone())?,
            ),
            _ => RegressionLearner::RandomForest(RandomForestConfig::from_r_list(r_list.clone())?),
        };

        Ok(Self {
            min_counts,
            min_cells,
            regression_learner,
            gene_batch_strategy,
            gene_batch_size,
            n_pcs,
            n_subsample,
        })
    }
}

//////////////////
// H5adFileTask //
//////////////////

impl H5adFileTask {
    /// Generate an H5adFileTask from an R list
    ///
    /// Expects: exp_id, h5_path, cs_type, no_cells, no_genes,
    /// gene_local_to_universe (integer vector, NA for unmapped genes,
    /// 0-indexed).
    pub fn from_r_list(r_list: List) -> extendr_api::Result<Self> {
        let map: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let raw_slot: RawDataSlot = match map.get("raw_slot").and_then(|v| v.as_str()) {
            Some(s) => {
                parse_raw_slot(s).ok_or_else(|| Error::Other(format!("Invalid raw slot: {}", s)))?
            }
            None => RawDataSlot::default(),
        };

        let exp_id = map
            .get("exp_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Other("exp_id missing or not a string".into()))?
            .to_string();

        let h5_path = map
            .get("h5_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Other("h5_path missing or not a string".into()))?
            .to_string();

        let cs_type_str = map
            .get("cs_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Other("cs_type missing or not a string".into()))?;
        let cs_type = parse_h5ad_format(cs_type_str)
            .ok_or_else(|| Error::Other("cs_type must be 'csr' or 'csc'".into()))?;

        let no_cells = map
            .get("no_cells")
            .and_then(|v| v.as_integer())
            .ok_or_else(|| Error::Other("no_cells missing or not an integer".into()))?
            as usize;

        let no_genes = map
            .get("no_genes")
            .and_then(|v| v.as_integer())
            .ok_or_else(|| Error::Other("no_genes missing or not an integer".into()))?
            as usize;

        let mapping_raw: Vec<i32> = map
            .get("gene_local_to_universe")
            .ok_or_else(|| Error::Other("gene_local_to_universe missing".into()))?
            .as_integer_slice()
            .ok_or_else(|| Error::Other("gene_local_to_universe must be an integer vector".into()))?
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

        Ok(Self {
            exp_id,
            h5_path,
            cs_type,
            no_cells,
            no_genes,
            gene_local_to_universe,
            raw_slot,
        })
    }
}

//////////////////
// BinMergeTask //
//////////////////

impl BinMergeTask {
    /// Generate a BinMergeTask from an R list
    ///
    /// Expects: exp_id, bin_cells_path, cells_to_keep (0-indexed integer
    /// vector), gene_local_to_universe (integer vector, NA for unmapped
    /// genes, 0-indexed).
    pub fn from_r_list(r_list: List) -> extendr_api::Result<Self> {
        let map: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let exp_id = map
            .get("exp_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Other("exp_id missing or not a string".into()))?
            .to_string();

        let bin_cells_path = map
            .get("bin_cells_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Other("bin_cells_path missing or not a string".into()))?
            .to_string();

        let cells_to_keep: Vec<usize> = map
            .get("cells_to_keep")
            .ok_or_else(|| Error::Other("cells_to_keep missing".into()))?
            .as_integer_slice()
            .ok_or_else(|| Error::Other("cells_to_keep must be an integer vector".into()))?
            .iter()
            .map(|&x| x as usize)
            .collect();

        let mapping_raw: Vec<i32> = map
            .get("gene_local_to_universe")
            .ok_or_else(|| Error::Other("gene_local_to_universe missing".into()))?
            .as_integer_slice()
            .ok_or_else(|| Error::Other("gene_local_to_universe must be an integer vector".into()))?
            .to_vec();

        // R NA_integer_ is i32::MIN
        let gene_local_to_universe: Vec<Option<u32>> = mapping_raw
            .into_iter()
            .map(|v| if v == i32::MIN { None } else { Some(v as u32) })
            .collect();

        Ok(Self {
            exp_id,
            bin_cells_path,
            cells_to_keep,
            gene_local_to_universe,
        })
    }
}

/////////////////
// MtxFileTask //
/////////////////

impl MtxFileTask {
    /// Generate an MtxFileTask from an R list
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list to converted to [MtxFileTask].
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn from_r_list(r_list: List) -> extendr_api::Result<Self> {
        let map: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let exp_id = map
            .get("exp_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Other("exp_id missing or not a string".into()))?
            .to_string();

        let mtx_path = map
            .get("mtx_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Other("mtx_path missing or not a string".into()))?
            .to_string();

        let cells_as_rows = map
            .get("cells_as_rows")
            .and_then(|v| v.as_bool())
            .ok_or_else(|| Error::Other("cells_as_rows missing or not a boolean".into()))?;

        let mapping_raw: Vec<i32> = map
            .get("gene_local_to_universe")
            .ok_or_else(|| Error::Other("gene_local_to_universe missing".into()))?
            .as_integer_slice()
            .ok_or_else(|| Error::Other("gene_local_to_universe must be an integer vector".into()))?
            .to_vec();

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

        Ok(Self {
            exp_id,
            mtx_path,
            gene_local_to_universe,
            cells_as_rows,
        })
    }
}

//////////////////
// TenxFileTask //
//////////////////

impl TenxFileTask {
    /// Generate a TenxFileTask from an R list
    ///
    /// Expects: exp_id, h5_path, version ("v2"/"v3"), no_cells, no_genes,
    /// gene_local_to_universe (integer vector, NA for unmapped / non-gene
    /// features, 0-indexed), feature_type (optional string).
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list to convert to [TenxFileTask].
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn from_r_list(r_list: List) -> extendr_api::Result<Self> {
        let map: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let exp_id = map
            .get("exp_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Other("exp_id missing or not a string".into()))?
            .to_string();
        let h5_path = map
            .get("h5_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Other("h5_path missing or not a string".into()))?
            .to_string();
        let version_str = map
            .get("version")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Other("version missing or not a string".into()))?;
        let version = parse_tenx_version(version_str)
            .ok_or_else(|| Error::Other("version must be 'v2' or 'v3'".into()))?;
        let no_cells = map
            .get("no_cells")
            .and_then(|v| v.as_integer())
            .ok_or_else(|| Error::Other("no_cells missing or not an integer".into()))?
            as usize;
        let no_genes = map
            .get("no_genes")
            .and_then(|v| v.as_integer())
            .ok_or_else(|| Error::Other("no_genes missing or not an integer".into()))?
            as usize;
        let mapping_raw: Vec<i32> = map
            .get("gene_local_to_universe")
            .ok_or_else(|| Error::Other("gene_local_to_universe missing".into()))?
            .as_integer_slice()
            .ok_or_else(|| Error::Other("gene_local_to_universe must be an integer vector".into()))?
            .to_vec();
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
        let feature_type = map
            .get("feature_type")
            .and_then(|v| v.as_str())
            .map(|v| v.to_string());
        Ok(Self {
            exp_id,
            h5_path,
            version,
            no_cells,
            no_genes,
            gene_local_to_universe,
            feature_type,
        })
    }
}

//////////////
// NicheNet //
//////////////

impl<T> LigandTargetParams<T>
where
    T: BixverseFloat,
{
    /// Generate the [LigandTargetParams] from an R list
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list to convert to [LigandTargetParams].
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn from_r_list(r_list: List) -> extendr_api::Result<Self> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = LigandTargetParams::default();

        let lr_sig_hub = params
            .get("lr_sig_hub")
            .and_then(|x| x.as_real())
            .and_then(|x| T::from_f64(x))
            .unwrap_or(defaults.lr_sig_hub);
        let gr_hub = params
            .get("gr_hub")
            .and_then(|x| x.as_real())
            .and_then(|x| T::from_f64(x))
            .unwrap_or(defaults.gr_hub);
        let ltf_cutoff = params
            .get("ltf_cutoff")
            .and_then(|x| x.as_real())
            .and_then(|x| T::from_f64(x))
            .unwrap_or(defaults.ltf_cutoff);
        let damping_factor = params
            .get("damping_factor")
            .and_then(|x| x.as_real())
            .and_then(|x| T::from_f64(x))
            .unwrap_or(defaults.damping_factor);
        let tol = params
            .get("tol")
            .and_then(|x| x.as_real())
            .and_then(|x| T::from_f64(x))
            .unwrap_or(defaults.tol);
        let max_iter = params
            .get("max_iter")
            .and_then(|x| x.as_integer())
            .map(|x| x as usize)
            .unwrap_or(defaults.max_iter);
        let topology_correction = params
            .get("topology_correction")
            .and_then(|x| x.as_bool())
            .unwrap_or(defaults.topology_correction);
        let secondary_targets = params
            .get("secondary_targets")
            .and_then(|x| x.as_bool())
            .unwrap_or(defaults.secondary_targets);

        Ok(Self {
            lr_sig_hub,
            gr_hub,
            ltf_cutoff,
            damping_factor,
            max_iter,
            tol,
            secondary_targets,
            topology_correction,
        })
    }
}

/////////////////////////
// MetaCells2 - Params //
/////////////////////////

//////////////////
// SelectParams //
//////////////////

impl SelectParams {
    /// Generate SelectParams from an R list, falling back to defaults.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = Self::default();

        let downsample_min_samples = params
            .get("downsample_min_samples")
            .and_then(|v| v.as_integer())
            .map(|v| v as u32)
            .unwrap_or(defaults.downsample_min_samples);

        let downsample_min_cell_quantile = params
            .get("downsample_min_cell_quantile")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.downsample_min_cell_quantile);

        let downsample_max_cell_quantile = params
            .get("downsample_max_cell_quantile")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.downsample_max_cell_quantile);

        let min_gene_total = params
            .get("min_gene_total")
            .and_then(|v| v.as_integer())
            .map(|v| Some(v as u32))
            .unwrap_or(defaults.min_gene_total);

        let min_gene_top3 = params
            .get("min_gene_top3")
            .and_then(|v| v.as_integer())
            .map(|v| Some(v as u32))
            .unwrap_or(defaults.min_gene_top3);

        let min_gene_relative_variance = params
            .get("min_gene_relative_variance")
            .and_then(|v| v.as_real())
            .map(|v| Some(v as f32))
            .unwrap_or(defaults.min_gene_relative_variance);

        let min_genes = params
            .get("min_genes")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.min_genes);

        let relative_variance_window_size = params
            .get("relative_variance_window_size")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.relative_variance_window_size);

        Ok(Self {
            downsample_min_samples,
            downsample_min_cell_quantile,
            downsample_max_cell_quantile,
            min_gene_total,
            min_gene_top3,
            min_gene_relative_variance,
            min_genes,
            relative_variance_window_size,
            lateral_gene_mask: defaults.lateral_gene_mask,
        })
    }
}

//////////////////////
// SimilarityParams //
//////////////////////

impl SimilarityParams {
    /// Generate SimilarityParams from an R list, falling back to defaults.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = Self::default();

        let method = match params.get("similarity_method").and_then(|v| v.as_str()) {
            Some(s) => match s.to_lowercase().as_str() {
                "log_pearson" | "logpearson" => SimilarityMethod::LogPearson,
                "pearson" => SimilarityMethod::Pearson,
                "spearman" => SimilarityMethod::Spearman,
                other => {
                    return Err(Error::Other(format!(
                        "Invalid similarity method: {}",
                        other
                    )));
                }
            },
            None => defaults.method,
        };

        let value_regularisation = params
            .get("value_regularisation")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.value_regularisation);

        Ok(Self {
            method,
            value_regularisation,
        })
    }
}

////////////////////
// MC2KnnParams   //
////////////////////

impl MC2KnnParams {
    /// Generate MC2KnnParams from an R list, falling back to defaults.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = Self::default();

        let balanced_ranks_factor = params
            .get("balanced_ranks_factor")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.balanced_ranks_factor);

        let incoming_degree_factor = params
            .get("incoming_degree_factor")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.incoming_degree_factor);

        let outgoing_degree_factor = params
            .get("outgoing_degree_factor")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.outgoing_degree_factor);

        let min_outgoing_degree = params
            .get("min_outgoing_degree")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.min_outgoing_degree);

        let k_size_factor = params
            .get("k_size_factor")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.k_size_factor);

        let k_umis_quantile = params
            .get("k_umis_quantile")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.k_umis_quantile);

        let min_knn_k = params
            .get("min_knn_k")
            .and_then(|v| v.as_integer())
            .map(|v| Some(v as usize))
            .unwrap_or(defaults.min_knn_k);

        let knn_k_override = params
            .get("knn_k_override")
            .and_then(|v| v.as_integer())
            .map(|v| Some(v as usize))
            .unwrap_or(defaults.knn_k_override);

        Ok(Self {
            balanced_ranks_factor,
            incoming_degree_factor,
            outgoing_degree_factor,
            min_outgoing_degree,
            k_size_factor,
            k_umis_quantile,
            min_knn_k,
            knn_k_override,
        })
    }
}

/////////////////////
// PartitionParams //
/////////////////////

impl PartitionParams {
    /// Generate PartitionParams from an R list, falling back to defaults.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = Self::default();

        let cooldown_pass = params
            .get("cooldown_pass")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.cooldown_pass);

        let cooldown_node = params
            .get("cooldown_node")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.cooldown_node);

        let cooldown_phase = params
            .get("cooldown_phase")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.cooldown_phase);

        let min_split_size_factor = params
            .get("min_split_size_factor")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.min_split_size_factor);

        let max_merge_size_factor = params
            .get("max_merge_size_factor")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.max_merge_size_factor);

        let max_split_min_cut_strength = params
            .get("max_split_min_cut_strength")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.max_split_min_cut_strength);

        let min_cut_seed_cells = params
            .get("min_cut_seed_cells")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.min_cut_seed_cells);

        let min_seed_size_quantile = params
            .get("min_seed_size_quantile")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.min_seed_size_quantile);

        let max_seed_size_quantile = params
            .get("max_seed_size_quantile")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.max_seed_size_quantile);

        Ok(Self {
            cooldown_pass,
            cooldown_node,
            cooldown_phase,
            min_split_size_factor,
            max_merge_size_factor,
            max_split_min_cut_strength,
            min_cut_seed_cells,
            min_seed_size_quantile,
            max_seed_size_quantile,
        })
    }
}

////////////////////
// DeviantsParams //
////////////////////

impl DeviantsParams {
    /// Generate DeviantsParams from an R list, falling back to defaults.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = Self::default();

        let min_gene_fold_factor = params
            .get("min_gene_fold_factor")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.min_gene_fold_factor);

        let max_gene_fraction = params
            .get("max_gene_fraction")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.max_gene_fraction);

        let max_cell_fraction = params
            .get("max_cell_fraction")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.max_cell_fraction);

        let gap_skip_cells = params
            .get("gap_skip_cells")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.gap_skip_cells);

        let max_gap_cells_count = params
            .get("max_gap_cells_count")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.max_gap_cells_count);

        let max_gap_cells_fraction = params
            .get("max_gap_cells_fraction")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.max_gap_cells_fraction);

        let min_compare_umis = params
            .get("min_compare_umis")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.min_compare_umis);

        let cells_regularisation_quantile = params
            .get("cells_regularisation_quantile")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.cells_regularisation_quantile);

        Ok(Self {
            min_gene_fold_factor,
            max_gene_fraction,
            max_cell_fraction,
            gap_skip_cells,
            max_gap_cells_count,
            max_gap_cells_fraction,
            min_compare_umis,
            cells_regularisation_quantile,
        })
    }
}

////////////////////
// DissolveParams //
////////////////////

impl DissolveParams {
    /// Generate DissolveParams from an R list, falling back to defaults.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = Self::default();

        let min_robust_size_factor = params
            .get("min_robust_size_factor")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.min_robust_size_factor);

        let min_convincing_gene_fold_factor = params
            .get("min_convincing_gene_fold_factor")
            .and_then(|v| v.as_real())
            .map(|v| Some(v as f32))
            .unwrap_or(defaults.min_convincing_gene_fold_factor);

        Ok(Self {
            min_robust_size_factor,
            min_convincing_gene_fold_factor,
        })
    }
}

/////////////////////
// MetacellsParams //
/////////////////////

impl MetacellsParams {
    /// Generate MetacellsParams from a flat R list, recursively filling each
    /// sub-struct from the same list. Falls back to defaults for any missing
    /// field.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let select = SelectParams::from_r_list(r_list.clone())?;
        let similarity = SimilarityParams::from_r_list(r_list.clone())?;
        let knn = MC2KnnParams::from_r_list(r_list.clone())?;
        let partition = PartitionParams::from_r_list(r_list.clone())?;
        let deviants = DeviantsParams::from_r_list(r_list.clone())?;
        let dissolve = DissolveParams::from_r_list(r_list.clone())?;

        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = Self::default();

        let target_metacell_size = params
            .get("target_metacell_size")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.target_metacell_size);

        let min_metacell_size = params
            .get("min_metacell_size")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.min_metacell_size);

        // u64 — read as real to dodge R's i32 limit.
        let target_metacell_umis = params
            .get("target_metacell_umis")
            .and_then(|v| v.as_real())
            .map(|v| v as u64)
            .unwrap_or(defaults.target_metacell_umis);

        let must_complete_cover = params
            .get("must_complete_cover")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.must_complete_cover);

        let random_seed = params
            .get("random_seed")
            .and_then(|v| v.as_real())
            .map(|v| v as u64)
            .unwrap_or(defaults.random_seed);

        Ok(Self {
            select,
            similarity,
            knn,
            partition,
            deviants,
            dissolve,
            target_metacell_size,
            min_metacell_size,
            target_metacell_umis,
            must_complete_cover,
            random_seed,
        })
    }
}

////////////
// ScType //
////////////

impl CellTypeMarkers {
    /// Generate a [CellTypeMarkers] from an R list
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list from which to parse the cell markers
    ///
    /// ### Returns
    ///
    /// The [CellTypeMarkers] or Error
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let data: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let cell_type = data
            .get("cell_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Other("cell_type missing or not a string".into()))?
            .to_string();

        let positive_indices: Vec<usize> = data
            .get("positive_indices")
            .and_then(|v| v.as_integer_vector())
            .map(|v| v.r_int_convert())
            .unwrap_or_default();

        let negative_indices: Vec<usize> = data
            .get("negative_indices")
            .and_then(|v| v.as_integer_vector())
            .map(|v| v.r_int_convert())
            .unwrap_or_default();

        Ok(Self {
            cell_type,
            positive_indices,
            negative_indices,
        })
    }
}

impl SctypeRes {
    /// Generate a [SctypeRes] from an R list
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list from which to parse the ScType results
    ///
    /// ### Returns
    ///
    /// The [SctypeRes] or Error
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let data: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let cell_types = data
            .get("cell_types")
            .and_then(|v| v.as_string_vector())
            .ok_or_else(|| Error::Other("cell_types missing or not a string vector".into()))?;

        let scores = data
            .get("scores")
            .and_then(|v| v.as_real_vector())
            .ok_or_else(|| Error::Other("scores missing or not a real vector".into()))?
            .r_float_convert();

        let n_cells: usize = data
            .get("n_cells")
            .and_then(|v| v.as_real())
            .map(|v| v as usize)
            .ok_or_else(|| Error::Other("n_cells missing or not an integer".into()))?;

        let n_cell_types: usize = data
            .get("n_cell_types")
            .and_then(|v| v.as_real())
            .map(|v| v as usize)
            .ok_or_else(|| Error::Other("n_cell_types missing or not an integer".into()))?;

        Ok(Self {
            cell_types,
            scores,
            n_cells,
            n_cell_types,
        })
    }
}

impl ScTypeCellParams {
    /// Generate the [ScTypeCellParams] from an R list
    ///
    /// Missing fields fall back to the defaults rather than erroring. An
    /// unrecognised calibration string is an error, since silently scoring on
    /// raw values would hide the mistake.
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list from which to parse the parameters
    ///
    /// ### Returns
    ///
    /// The [ScTypeCellParams] or Error
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let data: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let calibration =
            match data.get("calibration").and_then(|v| v.as_str()) {
                Some(s) => Some(parse_score_calibration(s).ok_or_else(|| {
                    Error::Other(format!("Invalid ScType score calibration: {}", s))
                })?),
                None => None,
            };

        Ok(Self::new(
            data.get("alpha")
                .and_then(|v| v.as_real())
                .map(|x| x as f32),
            data.get("iterations")
                .and_then(|v| v.as_integer())
                .map(|x| x as usize),
            data.get("tolerance")
                .and_then(|v| v.as_real())
                .map(|x| x as f32),
            calibration,
            data.get("score_floor")
                .and_then(|v| v.as_real())
                .map(|x| x as f32),
            data.get("purity_threshold")
                .and_then(|v| v.as_real())
                .map(|x| x as f32),
        ))
    }
}

//////////
// MELD //
//////////

impl MeldParams {
    /// Generate MeldParams from an R list, falling back to defaults.
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list from which to parse the arguments
    ///
    /// ### Returns
    ///
    /// The populated [MeldParams]
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let knn_params = KnnParams::from_r_list(r_list.clone())?;

        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = Self::default();

        let beta = params
            .get("beta")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.beta);

        let offset = params
            .get("offset")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.offset);

        let order = params
            .get("order")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.order);

        // An unrecognised string errors rather than falling back, matching
        // `ScTypeCellParams`: silently filtering with something other than what
        // was asked for hides the typo.
        let filter = match params.get("filter").and_then(|v| v.as_str()) {
            Some(s) => parse_meld_filter(s)
                .ok_or_else(|| Error::Other(format!("Invalid MELD filter: {}", s)))?,
            None => defaults.filter,
        };

        let chebyshev_order = params
            .get("chebyshev_order")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.chebyshev_order);

        let lap_type = match params.get("lap_type").and_then(|v| v.as_str()) {
            Some(s) => parse_lap_type(s)
                .ok_or_else(|| Error::Other(format!("Invalid MELD Laplacian type: {}", s)))?,
            None => defaults.lap_type,
        };

        let normalise_indicators = params
            .get("normalise_indicators")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.normalise_indicators);

        Ok(Self {
            beta,
            offset,
            order,
            filter,
            chebyshev_order,
            lap_type,
            normalise_indicators,
            knn_params,
        })
    }
}

impl AucellParams {
    /// Generate AucellParams from an R list, falling back to defaults.
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list from which to parse the arguments
    ///
    /// ### Returns
    ///
    /// The populated [AucellParams]
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = Self::default();

        let auc_type = match params.get("auc_type").and_then(|v| v.as_str()) {
            Some(s) => {
                parse_auc_type(s).ok_or_else(|| Error::Other(format!("Invalid AUC type: {}", s)))?
            }
            None => defaults.auc_type,
        };

        let max_rank = params
            .get("max_rank")
            .and_then(|v| v.as_real())
            .map(|v| v as usize)
            .or(defaults.max_rank);

        let standardise = params
            .get("standardise")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.standardise);

        Ok(Self {
            auc_type,
            max_rank,
            standardise,
        })
    }
}

impl BinariseParams {
    /// Generate BinariseParams from an R list, falling back to defaults.
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list from which to parse the arguments
    ///
    /// ### Returns
    ///
    /// The populated [BinariseParams]
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = Self::default();

        let bw_adjust = params
            .get("bw_adjust")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.bw_adjust);

        let n_grid = params
            .get("n_grid")
            .and_then(|v| v.as_real())
            .map(|v| v as usize)
            .unwrap_or(defaults.n_grid);

        let n_bins = params
            .get("n_bins")
            .and_then(|v| v.as_real())
            .map(|v| v as usize)
            .unwrap_or(defaults.n_bins);

        Ok(Self {
            bw_adjust,
            n_grid,
            n_bins,
        })
    }
}

//////////////
// Palantir //
//////////////

impl PalantirParams {
    /// Generate PalantirParams from an R list
    ///
    /// Any element missing from the list falls back to the matching field of
    /// [PalantirParams::default], so the two sides cannot drift apart. The
    /// nested kNN block is parsed by [KnnParams::from_r_list].
    ///
    /// Counts go through [r_list_count], which accepts both `10` and `10L` and
    /// errors on a value that is present but unusable. Previously a stray
    /// `knn = -5L` quietly became 30.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the Palantir parameters.
    ///
    /// ### Returns
    ///
    /// The `PalantirParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let defaults = Self::default();
        let knn_params = KnnParams::from_r_list(r_list.clone())?;

        let palantir_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let lanczos_params = LanczosParams::from_r_list(&palantir_list)?;

        let n_dcs = r_list_count(&palantir_list, "n_dcs")?.unwrap_or(defaults.n_dcs);

        let n_eigs = r_list_count(&palantir_list, "n_eigs")?;

        let knn = r_list_count(&palantir_list, "knn")?.unwrap_or(defaults.knn);

        let num_waypoints =
            r_list_count(&palantir_list, "num_waypoints")?.unwrap_or(defaults.num_waypoints);

        let scale_components = palantir_list
            .get("scale_components")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.scale_components);

        let use_early_cell_as_start = palantir_list
            .get("use_early_cell_as_start")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.use_early_cell_as_start);

        let max_iterations =
            r_list_count(&palantir_list, "max_iterations")?.unwrap_or(defaults.max_iterations);

        let branch_prob_threshold = palantir_list
            .get("branch_prob_threshold")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.branch_prob_threshold);

        Ok(Self {
            n_dcs,
            n_eigs,
            knn,
            num_waypoints,
            scale_components,
            use_early_cell_as_start,
            max_iterations,
            branch_prob_threshold,
            knn_params,
            lanczos_params,
        })
    }
}

///////////
// MAGIC //
///////////

impl MagicParams {
    /// Generate MagicParams from an R list
    ///
    /// Missing elements fall back to [MagicParams::default]. `layer` accepts
    /// `"norm"` or `"raw"`; anything else takes the default rather than
    /// erroring, matching how the other parsers here treat an unusable value.
    ///
    /// `n_steps` goes through [r_list_count_allow_zero] because zero is
    /// meaningful: it returns the un-imputed matrix.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the MAGIC parameters.
    ///
    /// ### Returns
    ///
    /// The `MagicParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let defaults = Self::default();
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let n_steps = r_list_count_allow_zero(&params, "n_steps")?.unwrap_or(defaults.n_steps);

        let clip_threshold = params
            .get("clip_threshold")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.clip_threshold);

        let gene_batch_size =
            r_list_count(&params, "gene_batch_size")?.unwrap_or(defaults.gene_batch_size);

        let layer = params
            .get("layer")
            .and_then(|v| v.as_str())
            .and_then(parse_magic_layer)
            .unwrap_or(defaults.layer);

        let allow_large = params
            .get("allow_large")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.allow_large);

        Ok(Self {
            n_steps,
            clip_threshold,
            gene_batch_size,
            layer,
            allow_large,
        })
    }
}

/// Parse the R-side layer string into a [MagicLayer].
///
/// ### Params
///
/// * `s` - The layer name, `"norm"` or `"raw"`, case-insensitive.
///
/// ### Returns
///
/// The matching layer, or `None` if the string is not recognised.
pub fn parse_magic_layer(s: &str) -> Option<MagicLayer> {
    match s.to_lowercase().as_str() {
        "norm" | "normalised" | "normalized" => Some(MagicLayer::Norm),
        "raw" | "counts" => Some(MagicLayer::Raw),
        _ => None,
    }
}

/////////////////
// Gene trends //
/////////////////

impl BranchSelectionParams {
    /// Generate BranchSelectionParams from an R list
    ///
    /// Missing elements fall back to [BranchSelectionParams::default], which
    /// carries the reference's `q = eps = 1e-2` and a resolution of 500.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the branch selection parameters.
    ///
    /// ### Returns
    ///
    /// The `BranchSelectionParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let defaults = Self::default();
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let q = params
            .get("q")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.q);

        let eps = params
            .get("eps")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.eps);

        let resolution = r_list_count(&params, "resolution")?.unwrap_or(defaults.resolution);

        Ok(Self { q, eps, resolution })
    }
}

impl LandmarkGpParams {
    /// Generate LandmarkGpParams from an R list
    ///
    /// Missing elements fall back to [LandmarkGpParams::default], which carries
    /// mellon's `length_scale = sigma = 1.0`. Those defaults are a strong
    /// smoothing prior; see the module documentation of
    /// [crate::single_cell::sc_trajectory::gene_trends].
    ///
    /// ### Params
    ///
    /// * `params` - Parsed R list contents, already flattened to a map. Shares
    ///   the list with [GeneTrendsParams], so the keys sit at the same level
    ///   rather than in a nested block.
    ///
    /// ### Returns
    ///
    /// The `LandmarkGpParams` with all parameters set.
    pub fn from_r_map(params: &HashMap<&str, Robj>) -> Result<Self> {
        let defaults = Self::default();

        let length_scale = params
            .get("length_scale")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.length_scale);

        let sigma = params
            .get("sigma")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.sigma);

        let jitter = params
            .get("jitter")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.jitter);

        let max_jitter_retries = r_list_count_allow_zero(params, "max_jitter_retries")?
            .unwrap_or(defaults.max_jitter_retries);

        // Zero is legal here and means "use the default", which is what
        // `fit_landmark_gp` does with it.
        let chunk_size =
            r_list_count_allow_zero(params, "chunk_size")?.unwrap_or(defaults.chunk_size);

        Ok(Self {
            length_scale,
            sigma,
            jitter,
            max_jitter_retries,
            chunk_size,
        })
    }
}

impl GeneTrendsParams {
    /// Generate GeneTrendsParams from an R list
    ///
    /// The Gaussian process hyperparameters live in the same flat list and are
    /// parsed by [LandmarkGpParams::from_r_map]. `weighting` accepts
    /// `"hard_mask"` or `"fate_probability"`.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the gene trend parameters.
    ///
    /// ### Returns
    ///
    /// The `GeneTrendsParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let defaults = Self::default();
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let gp = LandmarkGpParams::from_r_map(&params)?;

        let resolution = r_list_count(&params, "resolution")?.unwrap_or(defaults.resolution);

        let weighting = params
            .get("weighting")
            .and_then(|v| v.as_str())
            .and_then(parse_branch_weighting)
            .unwrap_or(defaults.weighting);

        Ok(Self {
            resolution,
            weighting,
            gp,
        })
    }
}

/// Parse the R-side weighting string into a [BranchWeighting].
///
/// ### Params
///
/// * `s` - The weighting name, `"hard_mask"` or `"fate_probability"`,
///   case-insensitive.
///
/// ### Returns
///
/// The matching weighting, or `None` if the string is not recognised.
pub fn parse_branch_weighting(s: &str) -> Option<BranchWeighting> {
    match s.to_lowercase().as_str() {
        "hard_mask" | "hardmask" | "mask" => Some(BranchWeighting::HardMask),
        "fate_probability" | "fateprobability" | "fate_prob" => {
            Some(BranchWeighting::FateProbability)
        }
        _ => None,
    }
}

//////////////
// DIALOGUE //
//////////////

impl PmdParams {
    /// Generate PmdParams from a flattened R list.
    ///
    /// Missing elements fall back to [PmdParams::default], which carries
    /// upstream's `DLG.get.param` values. Shares the list with [HlmParams] and
    /// [RefineParams], so the keys sit at the same level rather than in a
    /// nested block.
    ///
    /// ### Params
    ///
    /// * `params` - Parsed R list contents, already flattened to a map.
    ///
    /// ### Returns
    ///
    /// The `PmdParams` with all parameters set.
    pub fn from_r_map(params: &HashMap<&str, Robj>) -> Result<Self> {
        let defaults = Self::default();

        let k = r_list_count(params, "k")?.unwrap_or(defaults.k);

        let n_permutations =
            r_list_count(params, "n_permutations")?.unwrap_or(defaults.n_permutations);

        let extra_sparse = params
            .get("extra_sparse")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.extra_sparse);

        // Zero is legal and lets every sample into the ANOVA, however few cells
        // it contributed.
        let abn_c = r_list_count_allow_zero(params, "abn_c")?.unwrap_or(defaults.abn_c);

        let p_anova = params
            .get("p_anova")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.p_anova);

        let centre = params
            .get("centre")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.centre);

        let cap = params
            .get("cap")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.cap);

        let spatial = params
            .get("spatial")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.spatial);

        let n_genes = r_list_count(params, "n_genes")?.unwrap_or(defaults.n_genes);

        let min_ci = params
            .get("min_ci")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.min_ci);

        // An unrecognised string errors rather than falling back: collapsing
        // with something other than what was asked for hides the typo.
        let averaging = match params.get("averaging").and_then(|v| v.as_str()) {
            Some(s) => parse_averaging(s)
                .ok_or_else(|| Error::Other(format!("Invalid DIALOGUE averaging: {}", s)))?,
            None => defaults.averaging,
        };

        let mcp_assignment_p = params
            .get("mcp_assignment_p")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.mcp_assignment_p);

        // u64 — read as real to dodge R's i32 limit.
        let seed = params
            .get("seed")
            .and_then(|v| v.as_real())
            .map(|v| v as u64)
            .unwrap_or(defaults.seed);

        Ok(Self {
            k,
            n_permutations,
            extra_sparse,
            abn_c,
            p_anova,
            centre,
            cap,
            spatial,
            n_genes,
            min_ci,
            averaging,
            mcp_assignment_p,
            seed,
        })
    }
}

impl HlmParams {
    /// Generate HlmParams from a flattened R list.
    ///
    /// Missing elements fall back to [HlmParams::default]. Shares the list with
    /// [PmdParams] and [RefineParams].
    ///
    /// ### Params
    ///
    /// * `params` - Parsed R list contents, already flattened to a map.
    ///
    /// ### Returns
    ///
    /// The `HlmParams` with all parameters set.
    pub fn from_r_map(params: &HashMap<&str, Robj>) -> Result<Self> {
        let defaults = Self::default();

        // Zero is legal and drops the per-sample cell count requirement.
        let min_cells_per_sample = r_list_count_allow_zero(params, "min_cells_per_sample")?
            .unwrap_or(defaults.min_cells_per_sample);

        let use_tme_qc = params
            .get("use_tme_qc")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.use_tme_qc);

        let use_cell_quality = params
            .get("use_cell_quality")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.use_cell_quality);

        let satterthwaite = params
            .get("satterthwaite")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.satterthwaite);

        Ok(Self {
            min_cells_per_sample,
            use_tme_qc,
            use_cell_quality,
            satterthwaite,
        })
    }
}

impl RefineParams {
    /// Generate RefineParams from a flattened R list.
    ///
    /// Missing elements fall back to [RefineParams::default]. Shares the list
    /// with [PmdParams] and [HlmParams].
    ///
    /// ### Params
    ///
    /// * `params` - Parsed R list contents, already flattened to a map.
    ///
    /// ### Returns
    ///
    /// The `RefineParams` with all parameters set.
    pub fn from_r_map(params: &HashMap<&str, Robj>) -> Result<Self> {
        let defaults = Self::default();

        let support_p = params
            .get("support_p")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.support_p);

        let min_support_fraction = params
            .get("min_support_fraction")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.min_support_fraction);

        // Zero is legal and lets any non-empty stratum through.
        let min_stratum =
            r_list_count_allow_zero(params, "min_stratum")?.unwrap_or(defaults.min_stratum);

        let early_stop_cor = params
            .get("early_stop_cor")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.early_stop_cor);

        let permissive_p = params
            .get("permissive_p")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.permissive_p);

        let strict_p = params
            .get("strict_p")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.strict_p);

        Ok(Self {
            support_p,
            min_support_fraction,
            min_stratum,
            early_stop_cor,
            permissive_p,
            strict_p,
        })
    }
}

impl DialogueParams {
    /// Generate DialogueParams from a flat R list.
    ///
    /// The three stage blocks share one list, so the keys sit at the same level
    /// rather than in nested blocks. The list is flattened once and handed to
    /// [PmdParams::from_r_map], [HlmParams::from_r_map] and
    /// [RefineParams::from_r_map]. Nothing is validated here:
    /// [DialogueParams::validate] needs the cell type count, which the list does
    /// not carry, and `dialogue_run` calls it.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the DIALOGUE parameters.
    ///
    /// ### Returns
    ///
    /// The `DialogueParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        Ok(Self {
            pmd: PmdParams::from_r_map(&params)?,
            hlm: HlmParams::from_r_map(&params)?,
            refine: RefineParams::from_r_map(&params)?,
        })
    }
}

////////////
// NEBULA //
////////////

/// R-list parsing for [NebulaParams].
///
/// A trait rather than an inherent impl because [NebulaParams] is defined in
/// `edge-rs`.
pub trait NebulaParamsFromR: Sized {
    /// Parse the [NebulaParams] from a list
    ///
    /// ### Params
    ///
    /// * `params` - The flattened R list to parse
    ///
    /// ### Returns
    ///
    /// The [NebulaParams] populated by the R list, defaulting to the `nebula`
    /// package's own values.
    fn from_r_map(params: &HashMap<&str, Robj>) -> Result<Self>;
}

/// [NebulaParamsFromR] implementation
impl NebulaParamsFromR for NebulaParams {
    fn from_r_map(params: &HashMap<&str, Robj>) -> Result<Self> {
        let defaults = Self::default();

        let real = |key: &str, fallback: f64| -> f64 {
            params
                .get(key)
                .and_then(|v| v.as_real())
                .unwrap_or(fallback)
        };

        let method = match params.get("nebula_method").and_then(|v| v.as_str()) {
            Some(s) => parse_nebula_method(s)
                .ok_or_else(|| Error::Other(format!("Invalid NEBULA method: {}", s)))?,
            None => defaults.method,
        };

        Ok(Self {
            min: (
                real("min_sigma", defaults.min.0),
                real("min_phi", defaults.min.1),
            ),
            max: (
                real("max_sigma", defaults.max.0),
                real("max_phi", defaults.max.1),
            ),
            method,
            cutoff_cell: real("cutoff_cell", defaults.cutoff_cell),
            kappa: real("kappa", defaults.kappa),
            cpc: real("cpc", defaults.cpc),
            mincp: r_list_count(params, "mincp")?.unwrap_or(defaults.mincp),
            reml: params
                .get("reml")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.reml),
            eps: real("eps", defaults.eps),
        })
    }
}

/// R-list parsing for [ScTested].
///
/// A trait rather than an inherent impl because [ScTested] is defined in
/// `edge-rs`.
pub trait ScTestedFromR: Sized {
    /// Parse the [ScTested] from a list
    ///
    /// Expects either `coef`, a zero-based coefficient index, or `contrast`, one
    /// weight per coefficient. There is no default: which effect to report is
    /// the question the caller came to ask.
    ///
    /// ### Params
    ///
    /// * `params` - The flattened R list to parse
    ///
    /// ### Returns
    ///
    /// The [ScTested], or an error if neither key is present.
    fn from_r_map(params: &HashMap<&str, Robj>) -> Result<Self>;
}

/// [ScTestedFromR] implementation
impl ScTestedFromR for ScTested {
    fn from_r_map(params: &HashMap<&str, Robj>) -> Result<Self> {
        if let Some(contrast) = params.get("contrast").and_then(|v| v.as_real_vector()) {
            return Ok(ScTested::Contrast(contrast));
        }
        match r_list_count_allow_zero(params, "coef")? {
            Some(coef) => Ok(ScTested::Coef(coef)),
            None => Err(Error::Other(
                "NEBULA needs either `coef` or `contrast` to know what to test".into(),
            )),
        }
    }
}

impl NebulaScParams {
    /// Generate NebulaScParams from an R list
    ///
    /// The upstream NEBULA knobs fall back to the `nebula` package's own
    /// defaults. `coef` or `contrast` is required.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the NEBULA parameters.
    ///
    /// ### Returns
    ///
    /// The `NebulaScParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = Self::default();

        Ok(Self {
            nebula: NebulaParams::from_r_map(&params)?,
            gene_batch_size: r_list_count(&params, "gene_batch_size")?
                .unwrap_or(defaults.gene_batch_size),
            shrink_dispersion: params
                .get("shrink_dispersion")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.shrink_dispersion),
            tested: ScTested::from_r_map(&params)?,
        })
    }
}
