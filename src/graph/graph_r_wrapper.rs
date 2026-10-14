//! R wrappers over structures within the `graph` module

use extendr_api::*;

use crate::graph::leiden::LeidenParams;
use crate::prelude::*;

//////////////////
// LeidenParams //
//////////////////

impl LeidenParams {
    /// Parse the [LeidenParams] from an R list.
    ///
    /// Missing keys fall back to [LeidenParams::default]. Recognised keys:
    /// `resolution`, `beta`, `n_iterations`, `refine` and `seed`.
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list to parse
    ///
    /// ### Returns
    ///
    /// The [LeidenParams], or an error when a value is present but unusable.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params = r_list_to_map(r_list)?;
        let defaults = Self::default();

        Ok(Self {
            resolution: r_list_real(&params, "resolution")?.unwrap_or(defaults.resolution),
            beta: r_list_real(&params, "beta")?.unwrap_or(defaults.beta),
            n_iterations: r_list_count(&params, "n_iterations")?.unwrap_or(defaults.n_iterations),
            refine: params
                .get("refine")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.refine),
            seed: r_list_count_allow_zero(&params, "seed")?
                .map(|v| v as u64)
                .unwrap_or(defaults.seed),
        })
    }
}
