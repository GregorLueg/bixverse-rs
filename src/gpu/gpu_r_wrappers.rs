//! Contains R-specific functions for GPU-accelerated parts that need
//! the extendr interface.

use extendr_api::*;

use crate::gpu::sc_gpu::knn_gpu::CagraParams;

///////////
// CAGRA //
///////////

impl CagraParams {
    /// Generate the CagraParams params from an R list or default to sensible
    /// parameters
    ///
    /// ### Params
    ///
    /// * `r_list` - R list with the parameters
    ///
    /// ### Returns
    ///
    /// Self with the specified parameters.
    pub fn from_r_list(r_list: List) -> Self {
        let cagra = r_list.into_hashmap();

        let k = cagra.get("k").and_then(|v| v.as_integer()).unwrap_or(15) as usize;

        let ann_dist = std::string::String::from(
            cagra
                .get("ann_dist")
                .and_then(|v| v.as_str())
                .unwrap_or("cosine"),
        );

        let build_k_multiplier = cagra
            .get("build_k_multiplier")
            .and_then(|v| v.as_integer())
            .unwrap_or(2) as usize;

        let refine_sweeps = cagra
            .get("refine_sweeps")
            .and_then(|v| v.as_integer())
            .unwrap_or(1) as usize;

        let max_iters = cagra
            .get("max_iters")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        let n_trees = cagra
            .get("n_trees")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        let delta = cagra
            .get("delta")
            .and_then(|v| v.as_real())
            .unwrap_or(0.001) as f32;

        let rho = cagra.get("rho").and_then(|v| v.as_real()).map(|v| v as f32);

        let beam_width = cagra
            .get("beam_width")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        let max_beam_iters = cagra
            .get("max_beam_iters")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        let n_entry_points = cagra
            .get("n_entry_points")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        Self {
            k,
            ann_dist,
            build_k_multiplier,
            refine_sweeps,
            max_iters,
            n_trees,
            delta,
            rho,
            beam_width,
            max_beam_iters,
            n_entry_points,
        }
    }
}
