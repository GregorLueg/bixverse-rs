//! R methods that are involved in machine learning methods exposed in
//! `bixverse-rs`.

use extendr_api::*;
use std::collections::HashMap;

use crate::ml::clustering::k_means::*;

impl KMeansParamsWrappers {
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
            .unwrap_or(100) as usize;

        let init = params_list
            .get("k_means_init")
            .and_then(|v| v.as_str())
            .and_then(parse_kmeans_init);

        let gemm = params_list.get("gemm").and_then(|v| v.as_bool());
        let hamerly = params_list.get("hamerly").and_then(|v| v.as_bool());

        let path = match (gemm, hamerly) {
            (None, None) => None,
            (g, h) => Some(parse_kmean_path(g.unwrap_or(false), h.unwrap_or(false))),
        };

        Ok(Self::new(iters, init, path))
    }
}
