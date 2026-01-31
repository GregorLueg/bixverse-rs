//! Contains R-specific functions for single-cell data processing that need
//! the extendr interface.

use extendr_api::*;

use crate::single_cell::sc_data::data_io::MinCellQuality;

//////////////////
// Cell quality //
//////////////////

impl MinCellQuality {
    /// Generate the MinCellQuality params from an R list
    ///
    /// or default to sensible defaults
    ///
    /// ### Params
    ///
    /// * `r_list` - R list with the parameters
    ///
    /// ### Returns
    ///
    /// Self with the specified parameters.
    pub fn from_r_list(r_list: List) -> Self {
        let min_qc = r_list.into_hashmap();

        let min_unique_genes = min_qc
            .get("min_unique_genes")
            .and_then(|v| v.as_integer())
            .unwrap_or(100) as usize;

        let min_lib_size = min_qc
            .get("min_lib_size")
            .and_then(|v| v.as_integer())
            .unwrap_or(250) as usize;

        let min_cells = min_qc
            .get("min_cells")
            .and_then(|v| v.as_integer())
            .unwrap_or(10) as usize;

        let target_size = min_qc
            .get("target_size")
            .and_then(|v| v.as_real())
            .unwrap_or(1e5) as f32;

        MinCellQuality {
            min_unique_genes,
            min_lib_size,
            min_cells,
            target_size,
        }
    }
}
