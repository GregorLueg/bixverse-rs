//! Approximate nearest neighbour searches, accelerated via the wgpu backend on
//! the GPU.

use ann_search_rs::prelude::*;
use ann_search_rs::*;
use cubecl::wgpu::WgpuDevice;
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
pub fn cagra_knn_with_dist(
    embd: MatRef<f32>,
    cagra_params: &CagraParams,
    return_dist: bool,
    extract_knn: bool,
    seed: usize,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<f32>>>) {
    // helper
    fn remove_self(
        mut indices: Vec<Vec<usize>>,
        distances: Option<Vec<Vec<f32>>>,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<f32>>>) {
        for idx_vec in indices.iter_mut() {
            idx_vec.remove(0);
        }
        let distances = distances.map(|mut dists| {
            for dist_vec in dists.iter_mut() {
                dist_vec.remove(0);
            }
            dists
        });
        (indices, distances)
    }

    let start = Instant::now();
    let device: WgpuDevice = Default::default();
    let build_k = cagra_params.k * cagra_params.build_k_multiplier;

    if verbose {
        println!("Starting to generate the CAGRA index.")
    }

    let mut cagra_idx = build_nndescent_index_gpu::<f32, cubecl::wgpu::WgpuRuntime>(
        embd.as_ref(),
        &cagra_params.ann_dist,
        Some(cagra_params.k),
        Some(build_k),
        cagra_params.max_iters,
        cagra_params.n_trees,
        Some(cagra_params.delta),
        cagra_params.rho,
        Some(cagra_params.refine_sweeps),
        seed,
        false,
        true,
        device.clone(),
    );

    if verbose {
        println!("Generated the CAGRA index in {:.2?}.", start.elapsed());
    }

    let (indices, distances) = if extract_knn {
        if verbose {
            println!("Extracting the generated kNN graph directly.")
        }
        let (n, d) = extract_nndescent_knn_gpu(&cagra_idx, return_dist);
        if verbose {
            println!(" Extraction done in {:.2?}.", start.elapsed())
        }
        (n, d)
    } else {
        if verbose {
            println!("Generating the kNN graph via beam search.")
        }
        let search_params = CagraGpuSearchParams::new(
            cagra_params.beam_width,
            cagra_params.max_beam_iters,
            cagra_params.n_entry_points,
        );

        let (n, d) = query_nndescent_index_gpu_self(
            &mut cagra_idx,
            cagra_params.k + 1,
            Some(search_params),
            return_dist,
        );
        if verbose {
            println!(" Beam search done in {:.2?}.", start.elapsed())
        }
        (n, d)
    };

    remove_self(indices, distances)
}
