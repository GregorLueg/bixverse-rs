//! Contains R-specific functions for GPU-accelerated parts of this crate.

use extendr_api::*;
use std::collections::HashMap;

use super::k_means_gpu::KMeansGpuParams;
use crate::ml::clustering::k_means::parse_kmeans_init;

/////////////////////
// KMeansGpuParams //
/////////////////////

impl KMeansGpuParams {
    /// Parse the [KMeansParamsWrappers] from a list
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list to parse
    ///
    /// ### Returns
    ///
    /// The [KMeansParamsWrappers] populated by the R list.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params_list: HashMap<&str, Robj> = r_list.try_into()?;

        let iters = params_list
            .get("k_means_iter")
            .and_then(|v| v.as_integer())
            .unwrap_or(30) as usize;

        let init = params_list
            .get("k_means_init")
            .and_then(|v| v.as_str())
            .and_then(parse_kmeans_init);

        let tol = params_list
            .get("tol")
            .and_then(|v| v.as_real())
            .unwrap_or(1e-5);

        Ok(Self::new(iters, init, tol))
    }
}
