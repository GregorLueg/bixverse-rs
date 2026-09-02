//! Various functions, structures, etc. to expose more broadly when using this
//! crate in other libraries

pub use crate::core::math::sparse::{
    CompressedSparseData2, CompressedSparseFormat, LanczosParams, SparseAxis,
    parse_compressed_sparse_format,
};
pub use crate::core::math::vector_helpers::MAD_SCALE;
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
    RawCounts, SingleCellReading, from_cell_chunks, from_gene_chunks,
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

/// Step size, in percent, between progress reports.
///
/// Reporting on decile crossings rather than per unit of work bounds the number
/// of lines a sweep can emit at ten, no matter whether it walks ten blocks or
/// ten thousand genes.
const PROGRESS_STEP_PCT: usize = 10;

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

//////////////
// Progress //
//////////////

/// Prints a progress line whenever a sweep crosses a decile of its work.
///
/// Cheap enough to call on every unit of work: it formats nothing unless `done`
/// and `prev_done` fall either side of a [PROGRESS_STEP_PCT] boundary, or the
/// sweep has just finished. The verbosity check stays with the caller, since
/// not every sweep reports its progress at the same level.
///
/// ### Params
///
/// * `done` - Units of work finished, including the one just completed
/// * `prev_done` - Units of work finished before it
/// * `total` - Units of work in the whole sweep. Nothing prints if this is `0`
/// * `unit` - What is being counted, e.g. `"genes"`. Printed as given
/// * `elapsed` - Time since the sweep started
pub fn report_decile_progress(
    done: usize,
    prev_done: usize,
    total: usize,
    unit: &str,
    elapsed: std::time::Duration,
) {
    if total == 0 {
        return;
    }

    let pct = done * 100 / total;
    let prev_pct = prev_done * 100 / total;

    if pct / PROGRESS_STEP_PCT > prev_pct / PROGRESS_STEP_PCT || done == total {
        println!("  Progress: {pct}% ({done}/{total} {unit}, {elapsed:.2?})");
    }
}
