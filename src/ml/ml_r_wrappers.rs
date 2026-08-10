//! R methods that are involved in machine learning methods exposed in
//! `bixverse-rs`.

use extendr_api::*;
use std::collections::HashMap;

use crate::ml::clustering::k_means::*;
use crate::utils::r_rust_interface::{r_list_count, r_list_to_map};

impl KMeansParamsWrappers {
    /// Parse the [KMeansParamsWrappers] from a list
    ///
    /// A missing key falls back to the default. An unrecognised initialisation
    /// string is an error, since silently clustering with a different
    /// initialiser than the one asked for would hide the typo.
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list to parse
    ///
    /// ### Returns
    ///
    /// The [KMeansParamsWrappers] populated by the R list.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params_list: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let iters = r_list_count(&params_list, "k_means_iter")?.unwrap_or(30);

        let init =
            match params_list.get("k_means_init").and_then(|v| v.as_str()) {
                Some(s) => Some(parse_kmeans_init(s).ok_or_else(|| {
                    Error::Other(format!("Invalid k-means initialisation: {}", s))
                })?),
                None => None,
            };

        let gemm = params_list.get("gemm").and_then(|v| v.as_bool());
        let hamerly = params_list.get("hamerly").and_then(|v| v.as_bool());

        let path = match (gemm, hamerly) {
            (None, None) => None,
            (g, h) => Some(parse_kmean_path(g.unwrap_or(false), h.unwrap_or(false))),
        };

        Ok(Self::new(iters, init, path))
    }
}
