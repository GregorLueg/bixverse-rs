//! Various functions, structures, etc. to expose more broadly when using this
//! crate in other libraries

pub use crate::core::math::sparse::{
    CompressedSparseData2, CompressedSparseFormat, SparseAxis, parse_compressed_sparse_format,
};
pub use crate::errors::*;
pub use crate::graph::graph_structures::{EdgeData, NodeData, SparseGraph};
pub use crate::utils::heap_structures::RevOrderedFloat;
pub use crate::utils::matrix_utils::*;
pub use crate::utils::r_rust_interface::*;
pub use crate::utils::simd::BixverseSimd;
pub use crate::utils::traits::*;
pub use crate::utils::vec_utils::*;
pub use crate::{assert_nrows, assert_same_len, assert_symmetric_mat};

#[cfg(feature = "single-cell")]
pub use crate::single_cell::CELL_BATCH_SIZE;
#[cfg(feature = "single-cell")]
pub use crate::single_cell::sc_data::data_io::{
    CellQuality, CscGeneChunk, CsrCellChunk, DataLayerReturn, MinCellQuality, ParallelSparseReader,
    RawCounts, from_cell_chunks, from_gene_chunks,
};
#[cfg(feature = "single-cell")]
pub use crate::single_cell::sc_processing::knn::*;
#[cfg(feature = "single-cell")]
pub use crate::single_cell::sc_traits::*;

////////////
// Consts //
////////////

/// Version of the single cell files
#[cfg(feature = "single-cell")]
pub const SC_FILE_VERSION: u32 = 3;

/// Version of the meta cell files
#[cfg(feature = "single-cell")]
pub const MC_SPARSE_VERSION: u32 = 1;

///////////
// Enums //
///////////

/// Enum that controls verbosity
#[derive(Clone, Copy, Debug, Default)]
pub enum Verbosity {
    /// No verbosity at all
    #[default]
    Quiet,
    /// Normal levels of verbosity
    Normal,
    /// Detailed verbosity with increased messages
    Detailed,
}

impl Verbosity {
    /// Returns true if normal or detailed verbosity is set
    pub fn normal_verbosity(&self) -> bool {
        matches!(self, Verbosity::Normal | Verbosity::Detailed)
    }

    /// Returns true if detailed verbosity is set
    pub fn detailed_verbosity(&self) -> bool {
        matches!(self, Verbosity::Detailed)
    }
}

/// Parse verbosity leverl
///
/// ### Params
///
/// * `level` - If `1` returns [Verbosity::Normal], with `2`
///   [Verbosity::Detailed]
///
/// ### Returns
///
/// The desired [Verbosity] level.
pub fn parse_verbosity_level(level: usize) -> Verbosity {
    match level {
        0 => Verbosity::Quiet,
        1 => Verbosity::Normal,
        2 => Verbosity::Detailed,
        _ => Verbosity::Quiet,
    }
}
