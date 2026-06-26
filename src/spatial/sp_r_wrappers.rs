//! Contains R-specific functions for spatial data processing that need the
//! extendr interface.

use extendr_api::*;
use std::collections::HashMap;

use crate::spatial::sp_analysis::nhood_enrichment::NhoodEnrichmentParams;
use crate::spatial::sp_processing::sparkx::{SparkXKernel, SparkXParams};

////////////
// SparkX //
////////////

impl SparkXKernel {
    /// Generate a [SparkXKernel] from an R list
    ///
    /// ### Params
    ///
    /// * `r_list` - The list to convert into a Kernel
    ///
    /// ### Returns
    ///
    /// [SparkXKernel] if successful
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let map: HashMap<&str, Robj> = r_list.try_into()?;

        let bandwidth = map
            .get("bandwidth")
            .and_then(|v| v.as_real())
            .ok_or_else(|| {
                Error::Other(
                    "The provided bandwidth parameter could not be extracted correctly.".into(),
                )
            })? as f32;

        let kernel = map
            .get("kernel")
            .and_then(|x| x.as_str())
            .and_then(|x| match x {
                "gaussian" => Some(SparkXKernel::Gaussian { bandwidth }),
                "cosine" => Some(SparkXKernel::Cosine { bandwidth }),
                _ => None,
            })
            .ok_or_else(|| {
                Error::Other(
                    "The provided kernel parameter could not be extracted correctly.".into(),
                )
            })?;

        Ok(kernel)
    }
}

impl SparkXParams {
    /// Generate a [SparkXParams] from an R list
    ///
    /// ### Params
    ///
    /// * `r_list` - The list to convert into a Kernel
    ///
    /// ### Returns
    ///
    /// [SparkXParams] if successful
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list.try_into()?;
        let defaults = Self::default();

        let kernels = params
            .get("kernels")
            .and_then(|v| if v.is_null() { None } else { v.as_list() })
            .map(|list| {
                list.into_iter()
                    .map(|(_, robj)| {
                        List::try_from(robj)
                            .map_err(|e| Error::Other(e.into()))
                            .and_then(SparkXKernel::from_r_list)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();

        let n_landmarks = params
            .get("n_landmarks")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.n_landmarks);

        let bandwidth_subsample = params
            .get("bandwidth_subsample")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.bandwidth_subsample);

        Ok(Self {
            kernels,
            n_landmarks,
            bandwidth_subsample,
        })
    }
}

/////////////////////////////
// Nhood enrichment params //
/////////////////////////////

impl NhoodEnrichmentParams {
    /// Generate a [NhoodEnrichmentParams] from an R list
    ///
    /// ### Params
    ///
    /// * `r_list` - The list to convert into a Kernel
    ///
    /// ### Returns
    ///
    /// [NhoodEnrichmentParams] if successful
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list.try_into()?;
        let defaults = Self::default();

        let n_perm = params
            .get("n_perm")
            .and_then(|x| x.as_integer())
            .map(|x| x as usize)
            .unwrap_or(defaults.n_perm);

        let symmetrise = params
            .get("symmetrise")
            .and_then(|x| x.as_bool())
            .unwrap_or(defaults.symmetrise);

        Ok(Self { n_perm, symmetrise })
    }
}
