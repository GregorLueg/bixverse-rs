//! Contains R-specific functions for multi modal single cell data, especially
//! around extracting method-specific parameters from lists.

use extendr_api::{List, Robj};
use std::collections::HashMap;

use crate::single_cell::sc_processing::knn::KnnParams;
use crate::utils::r_rust_interface::r_list_to_map;
use crate::{
    prelude::VecConvert,
    single_cell::{multi_modal::wnn::parse_sigma_method, sc_processing::snn::parse_snn_type},
};

use super::{adt::dsb::DsbParams, wnn::WnnParams};

///////////////
// WnnParams //
///////////////

impl WnnParams {
    /// Generate WnnParams from a flat R list. Falls back to defaults for any
    /// missing field.
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list to parse
    ///
    /// ### Returns
    ///
    /// The [WnnParams]
    pub fn from_r_list(r_list: List) -> Result<Self, extendr_api::Error> {
        let knn_params = KnnParams::from_r_list(r_list.clone())?;
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = Self::default();

        let k_nn = params
            .get("k_nn")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.k_nn);

        let knn_range = params
            .get("knn_range")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.knn_range);

        let sigma_method = params
            .get("sigma_method")
            .and_then(|v| v.as_str())
            .and_then(parse_sigma_method)
            .unwrap_or(defaults.sigma_method);

        let sigma_idx = params
            .get("sigma_idx")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.sigma_idx);

        let snn_type = params
            .get("snn_type")
            .and_then(|v| v.as_str())
            .and_then(parse_snn_type)
            .unwrap_or(defaults.snn_type);

        let s_nn = params
            .get("s_nn")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.s_nn);

        let sd_scale = params
            .get("sd_scale")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.sd_scale);

        let kernel_power = params
            .get("kernel_power")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.kernel_power);

        let cross_const = params
            .get("cross_const")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.cross_const);

        let sigma_floor = params
            .get("sigma_floor")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.sigma_floor);

        Ok(Self {
            k_nn,
            knn_range,
            sigma_method,
            sigma_idx,
            snn_type,
            s_nn,
            sd_scale,
            kernel_power,
            cross_const,
            sigma_floor,
            knn_params,
        })
    }
}

///////////////
// DspParams //
///////////////

impl DsbParams {
    /// Generate the DsbParams from an R list + isotype indices
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list to parse
    /// * `isotype_indices` - The indices of the isotypes in the data
    ///
    /// ### Returns
    ///
    /// The [WnnParams]
    pub fn from_r_list(
        r_list: List,
        isotype_indices: Vec<i32>,
    ) -> Result<Self, extendr_api::Error> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = Self::default();

        let denoise_counts = params
            .get("denoise_counts")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.denoise_counts);

        let use_isotype_controls = params
            .get("use_isotype_controls")
            .and_then(|v| v.as_bool())
            .unwrap_or(defaults.use_isotype_controls);

        let isotype_indices = isotype_indices.r_int_convert();

        let pseudocount = params
            .get("pseudocount")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.pseudocount);

        let quantile_low = params.get("quantile_low").and_then(|v| v.as_real());
        let quantile_high = params.get("quantile_high").and_then(|v| v.as_real());
        let quantile_clip = match (quantile_low, quantile_high) {
            (Some(lo), Some(hi)) => Some((lo, hi)),
            _ => defaults.quantile_clip,
        };

        Ok(Self {
            denoise_counts,
            use_isotype_controls,
            isotype_indices,
            pseudocount,
            quantile_clip,
            ..defaults
        })
    }
}
