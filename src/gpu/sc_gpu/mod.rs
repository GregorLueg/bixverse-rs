//! Contains the single cell-specific GPU-accelerated methods. This includes
//! kNN graph generation and other aspects.

use ann_search_rs::*;
use cubecl::prelude::*;
use faer::MatRef;
use std::time::Instant;

////////////
// Params //
////////////

/// The parameters for the CAGRA-style kNN search
pub struct CagraParams {
    /// Number of neighbours to identify
    pub k: usize,
    /// Distance metric to use. One of `"euclidean"` or `"cosine"`.
    pub ann_dist: String,
    /// Multiplier for number of k-neighbours during build phase.
    pub build_k_multiplier: usize,
    /// Number of refinement sweeps during the generation of the
    pub refine_sweeps: usize,
    /// Maximum iterations for the NNDescent rounds
    pub max_iters: Option<usize>,
    /// Optional number of trees to use in the initial GPU-accelerated forest
    pub n_trees: Option<usize>,
    /// Termination criterium for the NNDescent iterations
    pub delta: f32,
    /// Optional sampling rate during NNDescent iterations.
    pub rho: Option<f32>,
    /// Beam width during querying
    pub beam_width: Option<usize>,
    /// Maximum beam iterations
    pub max_beam_iters: Option<usize>,
    /// Number of entry points into the graph
    pub n_entry_points: Option<usize>,
}

/// Use CAGRA-style GPU-accelerated kNN search.
///
/// Leverages the CAGRA style algorithm to generate a kNN graph from the data.
/// You have two options:
///
/// - Extract the graph directly after the NNDescent iterations and potential
///   refinement. Usually slightly lower precision, but faster.
/// - Run the beam search over the generated, pruned CAGRA graph for high
///   precison.
///
/// The algorithm runs on the wgpu backend.
///
/// ### Params
///
/// * `embd` - The embedding matrix to use to approximate neighbours and
///   calculate distances. Cells x features.
/// * `cagra_params` - Structure with the parameters for the CAGRA-style kNN
///   search.
/// * `return_dist` - Return the distances.
/// * `extract_knn` - Do you wish to use the fast extraction method.
/// * `seed` - Seed for reproducibility.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// Tuple of `(indices of nearest neighbours, distances to these neighbours)`
pub fn generate_knn_with_dist(
    embd: MatRef<f32>,
    cagra_params: &CagraParams,
    return_dist: bool,
    extract_knn: bool,
    seed: usize,
    verbose: bool,
) {
    let start = Instant::now();
    let device: cubecl::wgpu::WgpuDevice = Default::default();
    let build_k = cagra_params.k * cagra_params.build_k_multiplier;

    if verbose {
        println!("Starting to generate the CAGRA index.")
    }

    let mut cagra = build_nndescent_index_gpu::<f32, cubecl::wgpu::WgpuRuntime>(
        embd.as_ref(),
        &cagra_params.ann_dist,
        Some(cagra_params.k),
        Some(build_k),
        Some(20),
        None,
        Some(0.0005),
        None,
        Some(cagra_params.refine_sweeps),
        seed,
        false,
        true,
        device.clone(),
    );
}
