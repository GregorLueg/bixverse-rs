//! R methods that are involved in machine learning methods exposed in
//! `bixverse-rs`.

use extendr_api::*;
use std::collections::HashMap;

use crate::FromRList;
use crate::r_rust_interface::{r_list_count, r_list_real, r_list_to_map};
use bixverse_rs::ml::clustering::k_means::*;

//////////////////////////
// KMeansParamsWrappers //
//////////////////////////

/// Read the shared k-means keys from an already flattened R list.
///
/// Keys: `k_means_algorithm` (`"emeans"` or `"lloyd"`), `e_means_epochs`,
/// `k_means_iter`, `k_means_init`, `gemm` and `hamerly`. Names are checked
/// by [KMeansSpec]'s conversions, not here.
///
/// ### Params
///
/// * `params` - The flattened R list
///
/// ### Returns
///
/// The [KMeansSpec], or an error for a key of the wrong type.
pub fn kmeans_spec_from_r_map(params: &HashMap<&str, Robj>) -> Result<KMeansSpec> {
    let string = |key: &str| params.get(key).and_then(|v| v.as_str()).map(str::to_string);
    let flag = |key: &str| params.get(key).and_then(|v| v.as_bool());
    Ok(KMeansSpec {
        k_means_algorithm: string("k_means_algorithm"),
        e_means_epochs: r_list_real(params, "e_means_epochs")?,
        k_means_iter: r_list_count(params, "k_means_iter")?,
        k_means_init: string("k_means_init"),
        gemm: flag("gemm"),
        hamerly: flag("hamerly"),
    })
}

impl FromRList for KMeansParamsWrappers {
    /// Parse the [KMeansParamsWrappers] from a list, see
    /// [kmeans_spec_from_r_map].
    ///
    /// A missing key falls back to the default. An unrecognised algorithm or
    /// initialisation is an error, since silently clustering with something
    /// other than what was asked for would hide the typo.
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list to parse
    ///
    /// ### Returns
    ///
    /// The [KMeansParamsWrappers] populated by the R list.
    fn from_r_list(r_list: List) -> Result<Self> {
        let spec = kmeans_spec_from_r_map(&r_list_to_map(r_list)?)?;
        KMeansParamsWrappers::try_from(spec).map_err(Error::Other)
    }
}
